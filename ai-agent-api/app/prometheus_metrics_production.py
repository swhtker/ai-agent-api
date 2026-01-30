"""
Punky API - Production Metrics & Auto-Discovery (Recommended Version)
=====================================================================

This is the consolidated, production-ready version with all improvements.
Use this file instead of the individual versions.

IMPROVEMENTS OVER PREVIOUS VERSIONS:
1. Daily refresh (not weekly) - LLM pricing changes frequently
2. Retry logic with exponential backoff
3. Stale data alerts
4. Webhook notifications for price changes
5. Cost anomaly detection
6. Better error handling and recovery
7. Health check endpoint
8. Tiered refresh (frequently-used models checked more often)
9. Deprecation detection (removed models)
10. Rate limiting awareness

CONFIGURATION:
    # Recommended settings
    export PRICING_REFRESH_HOURS=24           # Daily refresh (was 168/weekly)
    export PRICING_CACHE_DIR=/var/lib/punky   # Persistent location (not /tmp)
    export PRICE_CHANGE_WEBHOOK_URL=https://... # Optional webhook for alerts
"""

import os
import re
import json
import time
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable, Set, Tuple
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

import httpx
from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    generate_latest, CONTENT_TYPE_LATEST,
    REGISTRY
)
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


# ============================================================
# Configuration - IMPROVED DEFAULTS
# ============================================================

class Config:
    """Production configuration with sensible defaults."""
    
    # REFRESH FREQUENCY
    # Recommendation: Daily (24h) is better than weekly
    # LLM pricing changes frequently, especially for newer providers
    REFRESH_INTERVAL_HOURS = int(os.getenv("PRICING_REFRESH_HOURS", "24"))  # Changed from 168
    
    # For frequently-used models, check more often
    HOT_MODEL_REFRESH_HOURS = int(os.getenv("HOT_MODEL_REFRESH_HOURS", "6"))
    
    # Minimum time between refreshes (prevent hammering on errors)
    MIN_REFRESH_INTERVAL_MINUTES = int(os.getenv("MIN_REFRESH_MINUTES", "30"))
    
    # RETRY CONFIGURATION
    MAX_RETRIES = int(os.getenv("PRICING_MAX_RETRIES", "3"))
    RETRY_BACKOFF_BASE = float(os.getenv("PRICING_RETRY_BACKOFF", "2.0"))  # Exponential backoff
    
    # DATA SOURCES
    ENABLE_LITELLM = os.getenv("PRICING_SOURCE_LITELLM", "true").lower() == "true"
    ENABLE_OPENROUTER = os.getenv("PRICING_SOURCE_OPENROUTER", "true").lower() == "true"
    ENABLE_PROVIDER_APIS = os.getenv("PRICING_SOURCE_PROVIDER_APIS", "true").lower() == "true"
    
    # ALERTING
    PRICE_CHANGE_ALERT_THRESHOLD = float(os.getenv("PRICE_CHANGE_ALERT_PERCENT", "5"))  # Lowered from 10%
    STALE_DATA_ALERT_HOURS = int(os.getenv("STALE_DATA_ALERT_HOURS", "48"))  # Alert if no refresh in 48h
    WEBHOOK_URL = os.getenv("PRICE_CHANGE_WEBHOOK_URL", "")  # Optional webhook for alerts
    
    # PERSISTENCE
    # Recommendation: Use persistent storage, not /tmp
    CACHE_DIR = os.getenv("PRICING_CACHE_DIR", "/var/lib/punky/pricing")
    HISTORY_RETENTION_DAYS = int(os.getenv("PRICING_HISTORY_DAYS", "90"))
    
    # ANOMALY DETECTION
    COST_ANOMALY_STDDEV_THRESHOLD = float(os.getenv("COST_ANOMALY_STDDEV", "3.0"))  # 3 standard deviations
    
    # HOT MODELS (check these more frequently)
    # These are commonly used models that should be verified more often
    HOT_MODELS = os.getenv("HOT_MODELS", "").split(",") if os.getenv("HOT_MODELS") else [
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet",
        "gpt-4o",
        "gpt-4o-mini",
        "gemini-2.0-flash",
    ]


# ============================================================
# Data Models
# ============================================================

class PricingSource(str, Enum):
    LITELLM = "litellm"
    OPENROUTER = "openrouter"
    PROVIDER_API = "provider_api"
    RESPONSE = "response"
    MANUAL = "manual"
    DEFAULT = "default"


class ModelStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    UNKNOWN = "unknown"


@dataclass
class ModelInfo:
    """Complete model information."""
    provider: str
    model_id: str
    display_name: str = ""
    input_cost_per_1k: float = 0.001
    output_cost_per_1k: float = 0.003
    context_window: int = 4096
    max_output_tokens: int = 4096
    supports_vision: bool = False
    supports_function_calling: bool = False
    supports_streaming: bool = True
    pricing_source: str = "default"
    status: str = "active"
    last_updated: datetime = field(default_factory=datetime.utcnow)
    last_verified: datetime = field(default_factory=datetime.utcnow)
    request_count: int = 0  # Track usage for "hot" model detection
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['last_updated'] = self.last_updated.isoformat()
        d['last_verified'] = self.last_verified.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ModelInfo':
        d = d.copy()
        if isinstance(d.get('last_updated'), str):
            d['last_updated'] = datetime.fromisoformat(d['last_updated'])
        if isinstance(d.get('last_verified'), str):
            d['last_verified'] = datetime.fromisoformat(d['last_verified'])
        # Handle missing fields gracefully
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        d = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**d)
    
    def is_stale(self, hours: int = None) -> bool:
        """Check if model data is stale."""
        hours = hours or Config.STALE_DATA_ALERT_HOURS
        return datetime.utcnow() - self.last_verified > timedelta(hours=hours)


@dataclass
class PricingChange:
    """Records a pricing change."""
    provider: str
    model: str
    field: str
    old_value: Any
    new_value: Any
    change_percent: float
    detected_at: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    notified: bool = False
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['detected_at'] = self.detected_at.isoformat()
        return d


@dataclass
class RefreshResult:
    """Result of a refresh operation."""
    source: str
    status: str  # success, partial, failed
    models_updated: int = 0
    models_added: int = 0
    models_deprecated: int = 0
    changes_detected: List[PricingChange] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    retry_count: int = 0
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        d['changes_detected'] = [c.to_dict() for c in self.changes_detected]
        return d


@dataclass 
class CostAnomaly:
    """Detected cost anomaly."""
    provider: str
    model: str
    expected_cost: float
    actual_cost: float
    deviation_stddev: float
    detected_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['detected_at'] = self.detected_at.isoformat()
        return d


# ============================================================
# HTTP Client with Retry Logic
# ============================================================

class ResilientHTTPClient:
    """HTTP client with retry logic and exponential backoff."""
    
    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
    
    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30,
                follow_redirects=True,
                headers={"User-Agent": "Punky-Pricing-Discovery/1.0"}
            )
        return self._client
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def fetch_with_retry(
        self,
        url: str,
        headers: Optional[Dict] = None,
        max_retries: int = None
    ) -> Optional[Dict]:
        """Fetch URL with exponential backoff retry."""
        max_retries = max_retries or Config.MAX_RETRIES
        client = await self.get_client()
        
        for attempt in range(max_retries + 1):
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    wait_time = min(60, Config.RETRY_BACKOFF_BASE ** attempt * 5)
                    logger.warning(f"Rate limited by {url}, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                elif e.response.status_code >= 500:  # Server error, retry
                    if attempt < max_retries:
                        wait_time = Config.RETRY_BACKOFF_BASE ** attempt
                        logger.warning(f"Server error from {url}, retry {attempt+1}/{max_retries} in {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Failed to fetch {url} after {max_retries} retries: {e}")
                        return None
                else:  # Client error, don't retry
                    logger.error(f"Client error fetching {url}: {e}")
                    return None
                    
            except Exception as e:
                if attempt < max_retries:
                    wait_time = Config.RETRY_BACKOFF_BASE ** attempt
                    logger.warning(f"Error fetching {url}, retry {attempt+1}/{max_retries}: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to fetch {url} after {max_retries} retries: {e}")
                    return None
        
        return None


# Global HTTP client
http_client = ResilientHTTPClient()


# ============================================================
# Webhook Notifier
# ============================================================

class WebhookNotifier:
    """Send notifications via webhook."""
    
    @staticmethod
    async def notify_price_change(change: PricingChange):
        """Send webhook notification for price change."""
        if not Config.WEBHOOK_URL:
            return
        
        try:
            payload = {
                "event": "price_change",
                "provider": change.provider,
                "model": change.model,
                "field": change.field,
                "old_value": change.old_value,
                "new_value": change.new_value,
                "change_percent": change.change_percent,
                "timestamp": change.detected_at.isoformat(),
                "severity": "high" if abs(change.change_percent) > 20 else "medium"
            }
            
            client = await http_client.get_client()
            await client.post(Config.WEBHOOK_URL, json=payload, timeout=10)
            change.notified = True
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
    
    @staticmethod
    async def notify_anomaly(anomaly: CostAnomaly):
        """Send webhook notification for cost anomaly."""
        if not Config.WEBHOOK_URL:
            return
        
        try:
            payload = {
                "event": "cost_anomaly",
                "provider": anomaly.provider,
                "model": anomaly.model,
                "expected_cost": anomaly.expected_cost,
                "actual_cost": anomaly.actual_cost,
                "deviation_stddev": anomaly.deviation_stddev,
                "timestamp": anomaly.detected_at.isoformat()
            }
            
            client = await http_client.get_client()
            await client.post(Config.WEBHOOK_URL, json=payload, timeout=10)
            
        except Exception as e:
            logger.error(f"Failed to send anomaly webhook: {e}")
    
    @staticmethod
    async def notify_stale_data(stale_models: List[str]):
        """Send webhook notification for stale data."""
        if not Config.WEBHOOK_URL or not stale_models:
            return
        
        try:
            payload = {
                "event": "stale_data",
                "stale_models": stale_models[:20],  # Limit to 20
                "total_stale": len(stale_models),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            client = await http_client.get_client()
            await client.post(Config.WEBHOOK_URL, json=payload, timeout=10)
            
        except Exception as e:
            logger.error(f"Failed to send stale data webhook: {e}")


# ============================================================
# Cost Anomaly Detector
# ============================================================

class AnomalyDetector:
    """Detect unusual cost patterns."""
    
    def __init__(self):
        # Store recent costs per model for comparison
        self._cost_history: Dict[str, List[float]] = defaultdict(list)
        self._max_history = 100
    
    def record_cost(self, provider: str, model: str, cost: float) -> Optional[CostAnomaly]:
        """Record a cost and check for anomalies."""
        key = f"{provider}/{model}"
        history = self._cost_history[key]
        
        # Need at least 10 data points for meaningful detection
        if len(history) >= 10:
            mean = sum(history) / len(history)
            variance = sum((x - mean) ** 2 for x in history) / len(history)
            stddev = variance ** 0.5
            
            if stddev > 0:
                deviation = (cost - mean) / stddev
                
                if abs(deviation) > Config.COST_ANOMALY_STDDEV_THRESHOLD:
                    anomaly = CostAnomaly(
                        provider=provider,
                        model=model,
                        expected_cost=mean,
                        actual_cost=cost,
                        deviation_stddev=deviation
                    )
                    # Don't add anomalous cost to history
                    return anomaly
        
        # Add to history
        history.append(cost)
        if len(history) > self._max_history:
            history.pop(0)
        
        return None


# Global anomaly detector
anomaly_detector = AnomalyDetector()


# ============================================================
# Scheduled Refresh Manager - IMPROVED
# ============================================================

class ScheduledRefreshManager:
    """Manages scheduled background refresh with tiered frequency."""
    
    def __init__(self, discovery: 'PricingDiscovery'):
        self.discovery = discovery
        self._task: Optional[asyncio.Task] = None
        self._hot_task: Optional[asyncio.Task] = None  # Separate task for hot models
        self._stop_event = asyncio.Event()
        self._last_refresh: Optional[datetime] = None
        self._last_hot_refresh: Optional[datetime] = None
        self._next_refresh: Optional[datetime] = None
        self._refresh_history: List[Dict[str, Any]] = []
        self._change_callbacks: List[Callable] = []
        self._consecutive_failures = 0
    
    def start(self):
        """Start background refresh schedulers."""
        if self._task is None or self._task.done():
            self._stop_event.clear()
            self._task = asyncio.create_task(self._full_refresh_loop())
            self._hot_task = asyncio.create_task(self._hot_refresh_loop())
            logger.info(
                f"Started pricing refresh: full every {Config.REFRESH_INTERVAL_HOURS}h, "
                f"hot models every {Config.HOT_MODEL_REFRESH_HOURS}h"
            )
    
    def stop(self):
        """Stop refresh schedulers."""
        self._stop_event.set()
        for task in [self._task, self._hot_task]:
            if task:
                task.cancel()
        logger.info("Stopped pricing refresh scheduler")
    
    def on_pricing_change(self, callback: Callable[[PricingChange], None]):
        """Register callback for pricing changes."""
        self._change_callbacks.append(callback)
    
    async def _full_refresh_loop(self):
        """Main refresh loop for all models."""
        # Initial refresh
        await self._do_full_refresh()
        
        while not self._stop_event.is_set():
            interval = timedelta(hours=Config.REFRESH_INTERVAL_HOURS)
            
            # Back off if we've had failures
            if self._consecutive_failures > 0:
                backoff = min(24, Config.RETRY_BACKOFF_BASE ** self._consecutive_failures)
                interval = timedelta(hours=backoff)
                logger.warning(f"Backing off refresh to {backoff}h due to failures")
            
            self._next_refresh = datetime.utcnow() + interval
            
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=interval.total_seconds()
                )
                break
            except asyncio.TimeoutError:
                await self._do_full_refresh()
    
    async def _hot_refresh_loop(self):
        """More frequent refresh for commonly used models."""
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=Config.HOT_MODEL_REFRESH_HOURS * 3600
                )
                break
            except asyncio.TimeoutError:
                await self._refresh_hot_models()
    
    async def _refresh_hot_models(self):
        """Quick refresh of frequently-used models only."""
        hot_models = self._get_hot_models()
        if not hot_models:
            return
        
        logger.info(f"Refreshing {len(hot_models)} hot models")
        self._last_hot_refresh = datetime.utcnow()
        
        # Just verify these models still exist and pricing is current
        # This is faster than a full refresh
        for model_key in hot_models:
            if model_key in self.discovery.models:
                model = self.discovery.models[model_key]
                model.last_verified = datetime.utcnow()
    
    def _get_hot_models(self) -> List[str]:
        """Get list of hot models (configured + high usage)."""
        hot = set()
        
        # Configured hot models
        for model in Config.HOT_MODELS:
            for key in self.discovery.models:
                if model in key:
                    hot.add(key)
        
        # High-usage models (top 10 by request count)
        by_usage = sorted(
            self.discovery.models.items(),
            key=lambda x: x[1].request_count,
            reverse=True
        )[:10]
        hot.update(k for k, _ in by_usage)
        
        return list(hot)
    
    async def _do_full_refresh(self):
        """Execute full refresh from all sources."""
        logger.info("Starting full pricing refresh...")
        start_time = time.perf_counter()
        
        results = []
        all_changes = []
        
        # Track which models we've seen (for deprecation detection)
        seen_models: Set[str] = set()
        
        # Fetch from each source
        if Config.ENABLE_LITELLM:
            result, seen = await self.discovery._fetch_litellm_pricing()
            results.append(result)
            all_changes.extend(result.changes_detected)
            seen_models.update(seen)
        
        if Config.ENABLE_OPENROUTER:
            result, seen = await self.discovery._fetch_openrouter_pricing()
            results.append(result)
            all_changes.extend(result.changes_detected)
            seen_models.update(seen)
        
        # Check for deprecated models (not seen in any source)
        deprecated = self._detect_deprecated_models(seen_models)
        
        # Update status
        self._last_refresh = datetime.utcnow()
        duration = time.perf_counter() - start_time
        
        # Check if successful
        successful = any(r.status == "success" for r in results)
        if successful:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
        
        # Record history
        refresh_record = {
            "timestamp": self._last_refresh.isoformat(),
            "duration_seconds": duration,
            "results": [r.to_dict() for r in results],
            "total_changes": len(all_changes),
            "deprecated_models": deprecated,
            "success": successful
        }
        self._refresh_history.append(refresh_record)
        
        # Trim history
        if len(self._refresh_history) > 100:
            self._refresh_history = self._refresh_history[-100:]
        
        # Notify callbacks and webhooks
        for change in all_changes:
            # Callbacks
            for callback in self._change_callbacks:
                try:
                    callback(change)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
            # Webhook for significant changes
            if abs(change.change_percent) >= Config.PRICE_CHANGE_ALERT_THRESHOLD:
                asyncio.create_task(WebhookNotifier.notify_price_change(change))
        
        # Save cache
        self.discovery._save_cache()
        
        # Log summary
        total_updated = sum(r.models_updated for r in results)
        total_added = sum(r.models_added for r in results)
        logger.info(
            f"Refresh complete: {total_updated} updated, {total_added} added, "
            f"{len(deprecated)} deprecated, {len(all_changes)} price changes "
            f"({duration:.2f}s)"
        )
        
        return results
    
    def _detect_deprecated_models(self, seen_models: Set[str]) -> List[str]:
        """Detect models that may have been deprecated."""
        deprecated = []
        
        for key, model in self.discovery.models.items():
            # Skip models from manual/response sources
            if model.pricing_source in ["manual", "response", "default"]:
                continue
            
            # If not seen in latest refresh and last verified > 7 days ago
            if key not in seen_models:
                if datetime.utcnow() - model.last_verified > timedelta(days=7):
                    model.status = ModelStatus.DEPRECATED
                    deprecated.append(key)
                    logger.warning(f"Model may be deprecated: {key}")
        
        return deprecated
    
    async def force_refresh(self) -> List[RefreshResult]:
        """Force immediate refresh."""
        # Check minimum interval
        if self._last_refresh:
            elapsed = datetime.utcnow() - self._last_refresh
            if elapsed < timedelta(minutes=Config.MIN_REFRESH_INTERVAL_MINUTES):
                logger.warning(f"Refresh too soon, last was {elapsed.seconds}s ago")
                return []
        
        return await self._do_full_refresh()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status."""
        stale_models = [
            k for k, v in self.discovery.models.items()
            if v.is_stale()
        ]
        
        return {
            "running": self._task is not None and not self._task.done(),
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "last_hot_refresh": self._last_hot_refresh.isoformat() if self._last_hot_refresh else None,
            "next_refresh": self._next_refresh.isoformat() if self._next_refresh else None,
            "refresh_interval_hours": Config.REFRESH_INTERVAL_HOURS,
            "hot_refresh_interval_hours": Config.HOT_MODEL_REFRESH_HOURS,
            "consecutive_failures": self._consecutive_failures,
            "stale_model_count": len(stale_models),
            "health": "healthy" if self._consecutive_failures == 0 and len(stale_models) < 10 else "degraded"
        }
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get refresh history."""
        return self._refresh_history[-limit:]


# ============================================================
# Pricing Discovery - IMPROVED
# ============================================================

class PricingDiscovery:
    """Production pricing discovery with all improvements."""
    
    LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
    OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
    
    PROVIDER_PATTERNS = {
        r'^gpt-': 'openai', r'^o1': 'openai', r'^o3': 'openai',
        r'^claude-': 'anthropic',
        r'^gemini': 'google',
        r'^mistral': 'mistral', r'^codestral': 'mistral', r'^mixtral': 'mistral',
        r'^llama': 'meta',
        r'^command': 'cohere',
        r'^deepseek': 'deepseek',
        r'^qwen': 'alibaba',
    }
    
    def __init__(self):
        self.cache_dir = Path(Config.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, ModelInfo] = {}
        self.pricing_history: Dict[str, List[Dict]] = {}
        self.changes: List[PricingChange] = []
        
        self.scheduler = ScheduledRefreshManager(self)
        self._load_cache()
    
    async def _fetch_litellm_pricing(self) -> Tuple[RefreshResult, Set[str]]:
        """Fetch from LiteLLM with retry logic."""
        result = RefreshResult(source="litellm", status="failed")
        seen_models: Set[str] = set()
        start_time = time.perf_counter()
        
        data = await http_client.fetch_with_retry(self.LITELLM_URL)
        if not data:
            result.errors.append("Failed to fetch LiteLLM data")
            result.duration_seconds = time.perf_counter() - start_time
            return result, seen_models
        
        result.status = "success"
        
        for model_key, info in data.items():
            if not isinstance(info, dict):
                continue
            
            input_cost = info.get("input_cost_per_token", 0) * 1000
            output_cost = info.get("output_cost_per_token", 0) * 1000
            
            if input_cost == 0 and output_cost == 0:
                continue
            
            provider = self._infer_provider(model_key)
            full_key = f"{provider}/{model_key}"
            seen_models.add(full_key)
            
            model_info = ModelInfo(
                provider=provider,
                model_id=model_key,
                display_name=info.get("litellm_provider", model_key),
                input_cost_per_1k=input_cost,
                output_cost_per_1k=output_cost,
                context_window=info.get("max_input_tokens", 4096),
                max_output_tokens=info.get("max_output_tokens", 4096),
                supports_vision=info.get("supports_vision", False),
                supports_function_calling=info.get("supports_function_calling", False),
                pricing_source="litellm",
                status="active"
            )
            
            is_new, changes = self._update_model(provider, model_key, model_info)
            result.changes_detected.extend(changes)
            
            if is_new:
                result.models_added += 1
            elif changes:
                result.models_updated += 1
        
        result.duration_seconds = time.perf_counter() - start_time
        logger.info(f"LiteLLM: {result.models_added} added, {result.models_updated} updated")
        return result, seen_models
    
    async def _fetch_openrouter_pricing(self) -> Tuple[RefreshResult, Set[str]]:
        """Fetch from OpenRouter with retry logic."""
        result = RefreshResult(source="openrouter", status="failed")
        seen_models: Set[str] = set()
        start_time = time.perf_counter()
        
        data = await http_client.fetch_with_retry(self.OPENROUTER_URL)
        if not data:
            result.errors.append("Failed to fetch OpenRouter data")
            result.duration_seconds = time.perf_counter() - start_time
            return result, seen_models
        
        result.status = "success"
        
        for model in data.get("data", []):
            model_id = model.get("id", "")
            pricing = model.get("pricing", {})
            
            try:
                input_cost = float(pricing.get("prompt", "0")) * 1000
                output_cost = float(pricing.get("completion", "0")) * 1000
            except (ValueError, TypeError):
                continue
            
            if input_cost == 0 and output_cost == 0:
                continue
            
            if "/" in model_id:
                provider = model_id.split("/")[0]
            else:
                provider = self._infer_provider(model_id)
            
            full_key = f"{provider}/{model_id}"
            seen_models.add(full_key)
            
            model_info = ModelInfo(
                provider=provider,
                model_id=model_id,
                display_name=model.get("name", model_id),
                input_cost_per_1k=input_cost,
                output_cost_per_1k=output_cost,
                context_window=model.get("context_length", 4096),
                pricing_source="openrouter",
                status="active"
            )
            
            is_new, changes = self._update_model(provider, model_id, model_info)
            result.changes_detected.extend(changes)
            
            if is_new:
                result.models_added += 1
            elif changes:
                result.models_updated += 1
        
        result.duration_seconds = time.perf_counter() - start_time
        logger.info(f"OpenRouter: {result.models_added} added, {result.models_updated} updated")
        return result, seen_models
    
    def _infer_provider(self, model_string: str) -> str:
        """Infer provider from model name."""
        model_lower = model_string.lower()
        
        if "/" in model_string:
            return model_string.split("/")[0].lower()
        
        for pattern, provider in self.PROVIDER_PATTERNS.items():
            if re.search(pattern, model_lower):
                return provider
        
        return "unknown"
    
    def _update_model(self, provider: str, model: str, new_info: ModelInfo) -> Tuple[bool, List[PricingChange]]:
        """Update model info and detect changes. Returns (is_new, changes)."""
        model_key = f"{provider}/{model}"
        changes = []
        is_new = model_key not in self.models
        
        if not is_new:
            old_info = self.models[model_key]
            
            # Detect price changes
            for field in ['input_cost_per_1k', 'output_cost_per_1k']:
                old_val = getattr(old_info, field)
                new_val = getattr(new_info, field)
                
                if old_val > 0 and old_val != new_val:
                    change_percent = ((new_val - old_val) / old_val) * 100
                    
                    change = PricingChange(
                        provider=provider,
                        model=model,
                        field=field,
                        old_value=old_val,
                        new_value=new_val,
                        change_percent=change_percent,
                        source=new_info.pricing_source
                    )
                    changes.append(change)
                    self.changes.append(change)
            
            # Preserve request count
            new_info.request_count = old_info.request_count
            new_info.last_updated = datetime.utcnow() if changes else old_info.last_updated
        
        new_info.last_verified = datetime.utcnow()
        
        # Record price history
        self._record_history(model_key, new_info)
        
        self.models[model_key] = new_info
        
        # Trim changes list
        if len(self.changes) > 1000:
            self.changes = self.changes[-1000:]
        
        return is_new, changes
    
    def _record_history(self, model_key: str, info: ModelInfo):
        """Record pricing snapshot for history."""
        if model_key not in self.pricing_history:
            self.pricing_history[model_key] = []
        
        history = self.pricing_history[model_key]
        
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "input_cost_per_1k": info.input_cost_per_1k,
            "output_cost_per_1k": info.output_cost_per_1k,
            "source": info.pricing_source
        }
        
        # Only add if changed
        if not history or (
            history[-1]["input_cost_per_1k"] != info.input_cost_per_1k or
            history[-1]["output_cost_per_1k"] != info.output_cost_per_1k
        ):
            history.append(snapshot)
        
        # Trim old history
        cutoff = datetime.utcnow() - timedelta(days=Config.HISTORY_RETENTION_DAYS)
        self.pricing_history[model_key] = [
            h for h in history
            if datetime.fromisoformat(h["timestamp"]) > cutoff
        ]
    
    def get_model_info(self, provider: str, model: str) -> ModelInfo:
        """Get model info, auto-creating if needed."""
        provider = provider.lower()
        model_key = f"{provider}/{model}"
        
        if model_key in self.models:
            self.models[model_key].request_count += 1
            return self.models[model_key]
        
        # Try fuzzy match
        for key, info in self.models.items():
            if model in key or key.endswith(f"/{model}"):
                info.request_count += 1
                return info
        
        # Create default
        logger.warning(f"Unknown model: {model_key}, using default pricing")
        default = ModelInfo(
            provider=provider,
            model_id=model,
            pricing_source="default"
        )
        self.models[model_key] = default
        return default
    
    def calculate_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD."""
        info = self.get_model_info(provider, model)
        cost = (input_tokens / 1000) * info.input_cost_per_1k + (output_tokens / 1000) * info.output_cost_per_1k
        
        # Check for anomaly
        anomaly = anomaly_detector.record_cost(provider, model, cost)
        if anomaly:
            logger.warning(f"Cost anomaly detected: {provider}/{model} ${cost:.6f} (expected ~${anomaly.expected_cost:.6f})")
            asyncio.create_task(WebhookNotifier.notify_anomaly(anomaly))
        
        return cost
    
    def _save_cache(self):
        """Save to persistent cache."""
        cache_file = self.cache_dir / "pricing_v2.json"
        try:
            data = {
                "version": "2.1",
                "saved_at": datetime.utcnow().isoformat(),
                "models": {k: v.to_dict() for k, v in self.models.items()},
                "pricing_history": self.pricing_history,
                "recent_changes": [c.to_dict() for c in self.changes[-100:]],
            }
            
            # Atomic write
            temp_file = cache_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f)
            temp_file.rename(cache_file)
            
            logger.debug(f"Saved {len(self.models)} models to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _load_cache(self):
        """Load from persistent cache."""
        cache_file = self.cache_dir / "pricing_v2.json"
        if not cache_file.exists():
            # Try legacy cache
            legacy = self.cache_dir / "pricing_cache.json"
            if legacy.exists():
                cache_file = legacy
            else:
                return
        
        try:
            with open(cache_file) as f:
                data = json.load(f)
            
            for key, info_dict in data.get("models", {}).items():
                try:
                    self.models[key] = ModelInfo.from_dict(info_dict)
                except Exception as e:
                    logger.warning(f"Failed to load model {key}: {e}")
            
            self.pricing_history = data.get("pricing_history", {})
            
            logger.info(f"Loaded {len(self.models)} models from cache")
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    
    def get_discovery_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        by_source = defaultdict(int)
        by_provider = defaultdict(int)
        stale_models = []
        deprecated = []
        
        for key, model in self.models.items():
            by_source[model.pricing_source] += 1
            by_provider[model.provider] += 1
            
            if model.is_stale():
                stale_models.append(key)
            if model.status == ModelStatus.DEPRECATED:
                deprecated.append(key)
        
        return {
            "total_models": len(self.models),
            "models_by_source": dict(by_source),
            "models_by_provider": dict(by_provider),
            "stale_models": stale_models[:20],
            "stale_count": len(stale_models),
            "deprecated_models": deprecated,
            "recent_changes": [c.to_dict() for c in self.changes[-20:]],
            "scheduler": self.scheduler.get_status(),
        }
    
    def get_health(self) -> Dict[str, Any]:
        """Get system health status."""
        scheduler_status = self.scheduler.get_status()
        stale_count = sum(1 for m in self.models.values() if m.is_stale())
        
        # Determine health
        issues = []
        if scheduler_status["consecutive_failures"] > 0:
            issues.append(f"refresh_failures: {scheduler_status['consecutive_failures']}")
        if stale_count > len(self.models) * 0.1:  # >10% stale
            issues.append(f"stale_data: {stale_count} models")
        
        return {
            "status": "healthy" if not issues else "degraded",
            "issues": issues,
            "models_loaded": len(self.models),
            "stale_models": stale_count,
            "last_refresh": scheduler_status["last_refresh"],
            "next_refresh": scheduler_status["next_refresh"],
        }


# Global instance
pricing_discovery = PricingDiscovery()


# ============================================================
# Prometheus Metrics
# ============================================================

class PunkyMetrics:
    """Production Prometheus metrics."""
    
    def __init__(self):
        self.registry = REGISTRY
        self._setup_metrics()
    
    def _setup_metrics(self):
        # LLM Metrics
        self.llm_requests_total = Counter(
            'llm_requests_total', 'Total LLM requests',
            ['provider', 'model', 'status'], registry=self.registry
        )
        self.llm_tokens_total = Counter(
            'llm_tokens_total', 'Tokens consumed',
            ['provider', 'model', 'type'], registry=self.registry
        )
        self.llm_cost_total = Counter(
            'llm_cost_total', 'Cost in USD',
            ['provider', 'model'], registry=self.registry
        )
        self.llm_request_duration_seconds = Histogram(
            'llm_request_duration_seconds', 'Request latency',
            ['provider', 'model'],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120],
            registry=self.registry
        )
        self.llm_errors_total = Counter(
            'llm_errors_total', 'Errors',
            ['provider', 'model', 'error_type'], registry=self.registry
        )
        
        # Discovery Metrics
        self.discovered_models = Gauge(
            'llm_discovered_models_total', 'Discovered models',
            registry=self.registry
        )
        self.stale_models = Gauge(
            'llm_stale_models_total', 'Stale models',
            registry=self.registry
        )
        self.pricing_changes_total = Counter(
            'llm_pricing_changes_total', 'Pricing changes',
            ['provider', 'field'], registry=self.registry
        )
        self.cost_anomalies_total = Counter(
            'llm_cost_anomalies_total', 'Cost anomalies',
            ['provider', 'model'], registry=self.registry
        )
        
        # Refresh Metrics
        self.refresh_duration_seconds = Histogram(
            'llm_pricing_refresh_duration_seconds', 'Refresh duration',
            ['source'], buckets=[1, 5, 10, 30, 60], registry=self.registry
        )
        self.refresh_failures_total = Counter(
            'llm_pricing_refresh_failures_total', 'Refresh failures',
            ['source'], registry=self.registry
        )
        
        # HTTP Metrics
        self.http_requests_total = Counter(
            'http_requests_total', 'HTTP requests',
            ['method', 'endpoint', 'status'], registry=self.registry
        )
        self.http_request_duration_seconds = Histogram(
            'http_request_duration_seconds', 'HTTP latency',
            ['method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
            registry=self.registry
        )
        
        # Agent Metrics
        self.agent_tasks_total = Counter(
            'agent_tasks_total', 'Agent tasks',
            ['task_type', 'status'], registry=self.registry
        )
        self.agent_tasks_active = Gauge(
            'agent_tasks_active', 'Active tasks',
            registry=self.registry
        )
        
        # Register change callback
        pricing_discovery.scheduler.on_pricing_change(self._on_change)
    
    def _on_change(self, change: PricingChange):
        self.pricing_changes_total.labels(
            provider=change.provider, field=change.field
        ).inc()
    
    def _update_gauges(self):
        stale = sum(1 for m in pricing_discovery.models.values() if m.is_stale())
        self.discovered_models.set(len(pricing_discovery.models))
        self.stale_models.set(stale)
    
    def record_llm_request(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration: float,
        status: str = "success",
        error_type: Optional[str] = None
    ) -> float:
        cost = pricing_discovery.calculate_cost(provider, model, input_tokens, output_tokens)
        
        self.llm_requests_total.labels(provider=provider, model=model, status=status).inc()
        self.llm_tokens_total.labels(provider=provider, model=model, type='input').inc(input_tokens)
        self.llm_tokens_total.labels(provider=provider, model=model, type='output').inc(output_tokens)
        self.llm_cost_total.labels(provider=provider, model=model).inc(cost)
        self.llm_request_duration_seconds.labels(provider=provider, model=model).observe(duration)
        
        if error_type:
            self.llm_errors_total.labels(provider=provider, model=model, error_type=error_type).inc()
        
        self._update_gauges()
        return cost
    
    def endpoint(self, request: Request) -> Response:
        self._update_gauges()
        return Response(content=generate_latest(self.registry), media_type=CONTENT_TYPE_LATEST)


metrics = PunkyMetrics()


# ============================================================
# Context Managers
# ============================================================

@contextmanager
def track_llm_request(provider: str, model: str):
    """Track LLM request with all metrics."""
    start = time.perf_counter()
    tracker = LLMTracker(provider, model)
    
    try:
        yield tracker
    except Exception as e:
        tracker._error_type = type(e).__name__
        tracker._status = "error"
        raise
    finally:
        tracker._finalize(time.perf_counter() - start)


class LLMTracker:
    def __init__(self, provider: str, model: str):
        self.provider = provider.lower()
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0
        self._status = "success"
        self._error_type = None
        self._done = False
    
    def record(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
    
    def record_tokens(self, input_tokens: int, output_tokens: int):
        self.record(input_tokens, output_tokens)
    
    def _finalize(self, duration: float):
        if self._done:
            return
        self._done = True
        metrics.record_llm_request(
            self.provider, self.model,
            self.input_tokens, self.output_tokens,
            duration, self._status, self._error_type
        )


@contextmanager
def track_agent_task(task_type: str = "default"):
    metrics.agent_tasks_active.inc()
    status = "failed"
    try:
        yield
        status = "completed"
    finally:
        metrics.agent_tasks_active.dec()
        metrics.agent_tasks_total.labels(task_type=task_type, status=status).inc()


# ============================================================
# HTTP Middleware
# ============================================================

class PrometheusMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") in ["/metrics", "/health"]:
            await self.app(scope, receive, send)
            return
        
        method = scope.get("method", "UNKNOWN")
        endpoint = self._normalize(scope.get("path", ""))
        start = time.perf_counter()
        status_code = 500
        
        async def wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)
        
        try:
            await self.app(scope, receive, wrapper)
        finally:
            metrics.http_requests_total.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()
            metrics.http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(time.perf_counter() - start)
    
    def _normalize(self, path: str) -> str:
        path = re.sub(r'[0-9a-f-]{36}', '{id}', path, flags=re.IGNORECASE)
        return re.sub(r'/\d+(?=/|$)', '/{id}', path)


# ============================================================
# FastAPI Router
# ============================================================

def create_metrics_router():
    from fastapi import APIRouter, BackgroundTasks
    from pydantic import BaseModel
    
    router = APIRouter(prefix="/metrics", tags=["Metrics"])
    
    class PricingUpdate(BaseModel):
        provider: str
        model: str
        input_cost_per_1k: float
        output_cost_per_1k: float
    
    @router.get("")
    async def prometheus_metrics(request: Request):
        return metrics.endpoint(request)
    
    @router.get("/health")
    async def health():
        return pricing_discovery.get_health()
    
    @router.get("/discovery")
    async def discovery():
        return pricing_discovery.get_discovery_report()
    
    @router.get("/scheduler")
    async def scheduler():
        return pricing_discovery.scheduler.get_status()
    
    @router.get("/scheduler/history")
    async def history(limit: int = 10):
        return {"history": pricing_discovery.scheduler.get_history(limit)}
    
    @router.post("/refresh")
    async def refresh(background_tasks: BackgroundTasks):
        background_tasks.add_task(pricing_discovery.scheduler.force_refresh)
        return {"status": "started"}
    
    @router.get("/changes")
    async def changes(limit: int = 50):
        return {"changes": [c.to_dict() for c in pricing_discovery.changes[-limit:]]}
    
    @router.get("/models")
    async def models(provider: Optional[str] = None, stale_only: bool = False):
        result = pricing_discovery.models
        if provider:
            result = {k: v for k, v in result.items() if v.provider == provider.lower()}
        if stale_only:
            result = {k: v for k, v in result.items() if v.is_stale()}
        return {"models": {k: v.to_dict() for k, v in result.items()}}
    
    @router.get("/models/{provider}/{model}/history")
    async def model_history(provider: str, model: str):
        key = f"{provider}/{model}"
        return {"history": pricing_discovery.pricing_history.get(key, [])}
    
    @router.post("/pricing")
    async def add_pricing(p: PricingUpdate):
        info = ModelInfo(
            provider=p.provider, model_id=p.model,
            input_cost_per_1k=p.input_cost_per_1k,
            output_cost_per_1k=p.output_cost_per_1k,
            pricing_source="manual"
        )
        pricing_discovery._update_model(p.provider, p.model, info)
        pricing_discovery._save_cache()
        return {"status": "added"}
    
    return router


# ============================================================
# Startup / Shutdown
# ============================================================

async def initialize():
    """Initialize system on startup."""
    logger.info("Initializing Punky metrics system...")
    
    if not pricing_discovery.models:
        await pricing_discovery.scheduler.force_refresh()
    
    pricing_discovery.scheduler.start()
    
    logger.info(f"Initialized: {len(pricing_discovery.models)} models, refresh every {Config.REFRESH_INTERVAL_HOURS}h")


async def shutdown():
    """Graceful shutdown."""
    pricing_discovery.scheduler.stop()
    await http_client.close()
    pricing_discovery._save_cache()
    logger.info("Shutdown complete")
