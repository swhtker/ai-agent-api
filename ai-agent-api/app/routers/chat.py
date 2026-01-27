from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import httpx
import os
import logging
import time

from ..telemetry import (
    get_tracer, 
    add_span_attributes, 
    record_exception,
    record_llm_tokens,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# RouteLLM Router endpoint
ROUTER_URL = os.getenv("ROUTER_URL", "http://routellm-router:8000/v1/chat/completions")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "router-mf-0.11593")


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


async def call_router(messages: List[dict]) -> dict:
    """Call the RouteLLM router to get a response."""
    tracer = get_tracer()
    
    with tracer.start_as_current_span("llm_router_call") as span:
        span.set_attribute("llm.router_url", ROUTER_URL)
        span.set_attribute("llm.model", ROUTER_MODEL)
        span.set_attribute("llm.message_count", len(messages))
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                logger.info(f"Calling router at {ROUTER_URL}")
                logger.info(f"Using model: {ROUTER_MODEL}")
                
                start_time = time.time()
                
                response = await client.post(
                    ROUTER_URL,
                    json={
                        "model": ROUTER_MODEL,
                        "messages": messages,
                    },
                    headers={"Content-Type": "application/json"},
                )
                
                duration = time.time() - start_time
                span.set_attribute("llm.duration_seconds", duration)
                
                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"Router error: {response.status_code} - {error_text}")
                    span.set_attribute("llm.error", True)
                    span.set_attribute("llm.status_code", response.status_code)
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Router error: {error_text}"
                    )
                
                result = response.json()
                
                # Extract token usage if available
                usage = result.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
                
                # Record metrics
                model_used = result.get("model", ROUTER_MODEL)
                span.set_attribute("llm.input_tokens", input_tokens)
                span.set_attribute("llm.output_tokens", output_tokens)
                span.set_attribute("llm.total_tokens", total_tokens)
                span.set_attribute("llm.model_used", model_used)
                
                # Record to metrics
                record_llm_tokens(input_tokens, output_tokens, model_used)
                
                logger.info(f"Router response received in {duration:.2f}s, tokens: {total_tokens}")
                
                return result
                
            except httpx.TimeoutException as e:
                logger.error(f"Router timeout: {e}")
                record_exception(e)
                raise HTTPException(status_code=504, detail="Router timeout")
            except httpx.RequestError as e:
                logger.error(f"Router request error: {e}")
                record_exception(e)
                raise HTTPException(status_code=502, detail=f"Router connection error: {str(e)}")


@router.post("/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """Handle chat completion requests via RouteLLM router."""
    tracer = get_tracer()
    
    with tracer.start_as_current_span("chat_completion") as span:
        span.set_attribute("chat.session_id", request.session_id or "none")
        span.set_attribute("chat.message_count", len(request.messages))
        
        try:
            # Convert messages to dict format
            messages = [{"role": m.role, "content": m.content} for m in request.messages]
            
            # Call the router
            result = await call_router(messages)
            
            # Extract response
            choices = result.get("choices", [])
            if not choices:
                raise HTTPException(status_code=500, detail="No response from router")
            
            response_content = choices[0].get("message", {}).get("content", "")
            model_used = result.get("model")
            
            # Get token count
            usage = result.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)
            
            span.set_attribute("chat.response_length", len(response_content))
            
            return ChatResponse(
                response=response_content,
                model_used=model_used,
                tokens_used=tokens_used,
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            record_exception(e)
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def chat_health():
    """Health check for chat router."""
    return {"status": "healthy", "router_url": ROUTER_URL}
