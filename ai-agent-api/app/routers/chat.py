"""
Punky API Chat Module - Fixed Version with EUR Cost Tracking
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import httpx
import os
import logging
import time
import asyncio
from datetime import datetime, timedelta, date

from ..telemetry import (
    get_tracer, 
    add_span_attributes, 
    record_exception,
    record_llm_tokens,
    record_request,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()

ROUTER_URL = os.getenv("ROUTER_URL", "http://routellm-router:8000/v1/chat/completions")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "router-mf-0.11593")

class ForexEODCache:
    """Cache for EUR/USD using EOD rates from last business day."""
    
    def __init__(self):
        self._rate = 0.92
        self._rate_date = None
        self._last_fetch = None
        self._cache_duration = timedelta(hours=4)
        self._lock = asyncio.Lock()
    
    async def get_eur_rate(self):
        async with self._lock:
            if self._should_refresh():
                await self._refresh_rate()
            return self._rate
    
    def _should_refresh(self):
        if self._last_fetch is None:
            return True
        return datetime.now() - self._last_fetch > self._cache_duration
    
    def _get_last_business_day(self):
        today = date.today()
        if today.weekday() == 0:
            return today - timedelta(days=3)
        elif today.weekday() >= 5:
            return today - timedelta(days=today.weekday()-4)
        return today - timedelta(days=1)
    
    async def _refresh_rate(self):
        eod_date = self._get_last_business_day()
        date_str = eod_date.strftime("%Y-%m-%d")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"https://api.frankfurter.app/{date_str}?from=USD&to=EUR")
                if resp.status_code == 200:
                    rate = resp.json().get("rates", {}).get("EUR")
                    if rate and 0.5 < rate < 2.0:
                        self._rate = rate
                        self._rate_date = eod_date
                        self._last_fetch = datetime.now()
                        logger.info(f"Forex EOD rate: 1 USD = {rate:.4f} EUR ({self._rate_date})")
                        return
        except Exception as e:
            logger.warning(f"Forex API failed: {e}")
    
    def get_rate_info(self):
        return {"rate": self._rate, "rate_date": str(self._rate_date) if self._rate_date else None}

forex_cache = ForexEODCache()

_prometheus_metrics = None

def _get_prometheus_metrics():
    global _prometheus_metrics
    if _prometheus_metrics is not None:
        return _prometheus_metrics
    try:
        from prometheus_client import Counter, Histogram
        metrics = {}
        try:
            metrics['requests'] = Counter('llm_requests_total', 'Total LLM requests', ['model', 'status'])
        except ValueError:
            pass
        try:
            metrics['tokens'] = Counter('llm_tokens_total', 'Tokens consumed', ['model', 'type'])
        except ValueError:
            pass
        try:
            metrics['cost_eur'] = Counter('llm_cost_eur_total', 'Cost in EUR', ['model'])
        except ValueError:
            pass
        try:
            metrics['errors'] = Counter('llm_errors_total', 'Errors', ['model', 'error_type'])
        except ValueError:
            pass
        try:
            metrics['duration'] = Histogram('llm_request_duration_seconds', 'Latency', ['model'])
        except ValueError:
            pass
        _prometheus_metrics = metrics
        return metrics
    except ImportError:
        _prometheus_metrics = {}
        return {}

def record_prometheus_metrics(model, input_tokens, output_tokens, duration, cost_eur, status="success"):
    metrics = _get_prometheus_metrics()
    if not metrics:
        return
    try:
        if 'requests' in metrics:
            metrics['requests'].labels(model=model, status=status).inc()
        if 'tokens' in metrics:
            metrics['tokens'].labels(model=model, type="input").inc(input_tokens)
            metrics['tokens'].labels(model=model, type="output").inc(output_tokens)
        if 'cost_eur' in metrics:
            metrics['cost_eur'].labels(model=model).inc(cost_eur)
        if 'duration' in metrics:
            metrics['duration'].labels(model=model).observe(duration)
    except Exception as e:
        logger.debug(f"Metrics error: {e}")

def record_error_metrics(model, error_type):
    metrics = _get_prometheus_metrics()
    if metrics:
        try:
            if 'requests' in metrics:
                metrics['requests'].labels(model=model, status="error").inc()
            if 'errors' in metrics:
                metrics['errors'].labels(model=model, error_type=error_type).inc()
        except:
            pass

MODEL_COSTS_USD_PER_1K = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "deepseek": {"input": 0.00014, "output": 0.00028},
    "default": {"input": 0.001, "output": 0.002},
}

async def calculate_cost_eur(model, input_tokens, output_tokens):
    rates = MODEL_COSTS_USD_PER_1K.get("default")
    model_lower = (model or "").lower()
    for key, model_rates in MODEL_COSTS_USD_PER_1K.items():
        if key != "default" and key in model_lower:
            rates = model_rates
            break
    cost_usd = (input_tokens / 1000) * rates["input"] + (output_tokens / 1000) * rates["output"]
    eur_rate = await forex_cache.get_eur_rate()
    return round(cost_usd * eur_rate, 8)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    cost_eur: Optional[float] = None

async def call_router(messages):
    tracer = get_tracer()
    with tracer.start_as_current_span("llm_router_call") as span:
        span.set_attribute("llm.router_url", ROUTER_URL)
        span.set_attribute("llm.model", ROUTER_MODEL)
        start_time = time.time()
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(ROUTER_URL, json={"model": ROUTER_MODEL, "messages": messages}, headers={"Content-Type": "application/json"})
                duration = time.time() - start_time
                if response.status_code != 200:
                    record_error_metrics(ROUTER_MODEL, f"http_{response.status_code}")
                    raise HTTPException(status_code=response.status_code, detail=response.text)
                result = response.json()
                usage = result.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                model_used = result.get("model", ROUTER_MODEL)
                cost_eur = await calculate_cost_eur(model_used, input_tokens, output_tokens)
                span.set_attribute("llm.cost_eur", cost_eur)
                record_llm_tokens(input_tokens, output_tokens, model_used)
                record_request()
                record_prometheus_metrics(model_used, input_tokens, output_tokens, duration, cost_eur)
                result["_cost_eur"] = cost_eur
                logger.info(f"LLM: model={model_used}, tokens={input_tokens+output_tokens}, cost=EUR{cost_eur:.6f}")
                return result
            except httpx.TimeoutException as e:
                record_exception(e)
                record_error_metrics(ROUTER_MODEL, "timeout")
                raise HTTPException(status_code=504, detail="Router timeout")
            except httpx.RequestError as e:
                record_exception(e)
                record_error_metrics(ROUTER_MODEL, "connection_error")
                raise HTTPException(status_code=502, detail=str(e))

@router.post("/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    tracer = get_tracer()
    with tracer.start_as_current_span("chat_completion") as span:
        span.set_attribute("chat.session_id", request.session_id or "none")
        try:
            messages = [{"role": m.role, "content": m.content} for m in request.messages]
            result = await call_router(messages)
            choices = result.get("choices", [])
            if not choices:
                raise HTTPException(status_code=500, detail="No response")
            return ChatResponse(
                response=choices[0].get("message", {}).get("content", ""),
                model_used=result.get("model"),
                tokens_used=result.get("usage", {}).get("total_tokens", 0),
                cost_eur=result.get("_cost_eur", 0.0),
            )
        except HTTPException:
            raise
        except Exception as e:
            record_exception(e)
            record_error_metrics(ROUTER_MODEL, "internal_error")
            raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def chat_health():
    return {"status": "healthy", "router_url": ROUTER_URL, "forex": forex_cache.get_rate_info()}

@router.get("/forex-rate")
async def get_forex_rate():
    rate = await forex_cache.get_eur_rate()
    return {"usd_to_eur": rate, **forex_cache.get_rate_info()}
