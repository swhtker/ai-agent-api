"""
Punky Agent Telemetry - Enhanced Module
Autonomous agent observability for competing with Perplexity Comet & Claude Cowork.
"""

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from functools import wraps

from opentelemetry import trace, metrics

try:
    from .telemetry import telemetry, OTEL_ENABLED, SERVICE_NAME_VALUE
except ImportError:
    from telemetry import telemetry, OTEL_ENABLED, SERVICE_NAME_VALUE

logger = logging.getLogger(__name__)

COMPETITOR_COSTS = {
    "perplexity_pro": 0.015,
    "claude_cowork": 0.012,
    "chatgpt_plus": 0.020,
    "default": 0.015,
}

@dataclass
class TaskContext:
    task_id: str
    task_type: str
    start_time: float = field(default_factory=time.time)
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    steps: int = 0
    tools_used: List[str] = field(default_factory=list)
    llm_calls: int = 0
    retries: int = 0
    providers_used: List[str] = field(default_factory=list)
    failovers: int = 0
    max_context_used: int = 0
    context_window_size: int = 128000
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output
    
    @property
    def duration_seconds(self) -> float:
        return time.time() - self.start_time
    
    @property
    def context_utilization(self) -> float:
        if self.context_window_size == 0:
            return 0.0
        return self.max_context_used / self.context_window_size
    
    def add_tokens(self, input_tokens: int, output_tokens: int):
        self.tokens_input += input_tokens
        self.tokens_output += output_tokens
    
    def add_cost(self, cost: float):
        self.cost_usd += cost
    
    def increment_steps(self):
        self.steps += 1
    
    def add_tool(self, tool_name: str):
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)
    
    def add_llm_call(self, provider: str = None):
        self.llm_calls += 1
        if provider and provider not in self.providers_used:
            self.providers_used.append(provider)
    
    def record_retry(self):
        self.retries += 1
    
    def record_failover(self):
        self.failovers += 1

class AgentTelemetry:
    def __init__(self):
        self._initialized = False
        self._metrics = None
        self._tracer = None
    
    def _ensure_initialized(self):
        if self._initialized:
            return
        if not OTEL_ENABLED:
            logger.info("Agent telemetry disabled (OTEL_ENABLED=false)")
            self._initialized = True
            return
        try:
            self._init_metrics()
            self._tracer = trace.get_tracer(f"{SERVICE_NAME_VALUE}-agent")
            self._initialized = True
            logger.info("Agent telemetry initialized")
        except Exception as e:
            logger.error(f"Failed to initialize agent telemetry: {e}")
            self._initialized = True
    
    def _init_metrics(self):
        meter = metrics.get_meter(f"{SERVICE_NAME_VALUE}-agent")
        self._metrics = {
            "tasks_total": meter.create_counter("punky_agent_tasks_total", description="Total agent tasks", unit="1"),
            "task_duration": meter.create_histogram("punky_agent_task_duration_seconds", description="Task duration", unit="s"),
            "task_cost": meter.create_histogram("punky_agent_task_cost_usd", description="Cost per task", unit="1"),
            "task_steps": meter.create_histogram("punky_agent_task_steps", description="Steps per task", unit="1"),
            "task_tokens": meter.create_histogram("punky_agent_task_tokens_total", description="Tokens per task", unit="1"),
            "task_llm_calls": meter.create_histogram("punky_agent_task_llm_calls", description="LLM calls per task", unit="1"),
            "tool_executions": meter.create_counter("punky_tool_executions_total", description="Tool executions", unit="1"),
            "tool_duration": meter.create_histogram("punky_tool_duration_seconds", description="Tool duration", unit="s"),
            "tool_errors": meter.create_counter("punky_tool_errors_total", description="Tool errors", unit="1"),
            "provider_requests": meter.create_counter("punky_provider_requests_total", description="Provider requests", unit="1"),
            "provider_failovers": meter.create_counter("punky_provider_failovers_total", description="Failovers", unit="1"),
            "provider_latency": meter.create_histogram("punky_provider_latency_seconds", description="Provider latency", unit="s"),
            "cost_savings": meter.create_counter("punky_cost_savings_usd_total", description="Cost savings vs competitors", unit="1"),
            "cost_by_feature": meter.create_counter("punky_cost_by_feature_usd_total", description="Cost by feature", unit="1"),
            "context_utilization": meter.create_histogram("punky_context_utilization_ratio", description="Context utilization", unit="1"),
            "context_truncations": meter.create_counter("punky_context_truncations_total", description="Context truncations", unit="1"),
            "task_retries": meter.create_counter("punky_agent_task_retries_total", description="Task retries", unit="1"),
            "active_tasks": meter.create_up_down_counter("punky_agent_active_tasks", description="Active tasks", unit="1"),
        }

    @contextmanager
    def track_task(self, task_id: str = None, task_type: str = "generic"):
        self._ensure_initialized()
        task_id = task_id or str(uuid.uuid4())[:8]
        ctx = TaskContext(task_id=task_id, task_type=task_type)
        status = "success"
        
        if self._metrics:
            self._metrics["active_tasks"].add(1, {"task_type": task_type})
        
        span = self._tracer.start_span(f"agent_task_{task_type}") if self._tracer else None
        if span:
            span.set_attribute("task.id", task_id)
            span.set_attribute("task.type", task_type)
        
        try:
            yield ctx
        except Exception as e:
            status = "failed"
            if span:
                span.record_exception(e)
            raise
        finally:
            duration = ctx.duration_seconds
            if span:
                span.set_attribute("task.duration_seconds", duration)
                span.set_attribute("task.cost_usd", ctx.cost_usd)
                span.set_attribute("task.tokens_total", ctx.total_tokens)
                span.set_attribute("task.status", status)
                span.end()
            
            if self._metrics:
                attrs = {"task_type": task_type, "status": status}
                self._metrics["tasks_total"].add(1, attrs)
                self._metrics["active_tasks"].add(-1, {"task_type": task_type})
                self._metrics["task_duration"].record(duration, {"task_type": task_type})
                self._metrics["task_cost"].record(ctx.cost_usd, {"task_type": task_type})
                self._metrics["task_steps"].record(ctx.steps, {"task_type": task_type})
                self._metrics["cost_by_feature"].add(ctx.cost_usd, {"feature": task_type})
                
                competitor_cost = self._estimate_competitor_cost(ctx.total_tokens)
                savings = max(0, competitor_cost - ctx.cost_usd)
                if savings > 0:
                    self._metrics["cost_savings"].add(savings)
            
            logger.info(f"Task {task_id}: {status} in {duration:.2f}s, cost=${ctx.cost_usd:.6f}")
    
    @contextmanager
    def track_tool(self, tool_name: str, task_ctx: TaskContext = None):
        self._ensure_initialized()
        start = time.time()
        status = "success"
        span = self._tracer.start_span(f"tool_{tool_name}") if self._tracer else None
        
        try:
            yield
            if task_ctx:
                task_ctx.add_tool(tool_name)
        except Exception as e:
            status = "failed"
            if span:
                span.record_exception(e)
            if self._metrics:
                self._metrics["tool_errors"].add(1, {"tool": tool_name, "error_type": type(e).__name__})
            raise
        finally:
            duration = time.time() - start
            if span:
                span.set_attribute("tool.duration_seconds", duration)
                span.end()
            if self._metrics:
                self._metrics["tool_executions"].add(1, {"tool": tool_name, "status": status})
                self._metrics["tool_duration"].record(duration, {"tool": tool_name})

    def record_provider_request(self, provider: str, model: str, latency_seconds: float, success: bool, task_ctx: TaskContext = None):
        self._ensure_initialized()
        if task_ctx:
            task_ctx.add_llm_call(provider)
        if self._metrics:
            self._metrics["provider_requests"].add(1, {"provider": provider, "model": model, "success": str(success).lower()})
            self._metrics["provider_latency"].record(latency_seconds, {"provider": provider, "model": model})
    
    def record_failover(self, from_provider: str, to_provider: str, reason: str, task_ctx: TaskContext = None):
        self._ensure_initialized()
        if task_ctx:
            task_ctx.record_failover()
        if self._metrics:
            self._metrics["provider_failovers"].add(1, {"from_provider": from_provider, "to_provider": to_provider, "reason": reason})
        logger.warning(f"Provider failover: {from_provider} -> {to_provider} ({reason})")
    
    def record_context_truncation(self, model: str, original_tokens: int, truncated_to: int):
        self._ensure_initialized()
        if self._metrics:
            self._metrics["context_truncations"].add(1, {"model": model})
        logger.info(f"Context truncated: {original_tokens} -> {truncated_to} for {model}")

        def record_chat_task(
                    self,
                    model: str = "unknown",
                    input_tokens: int = 0,
                    output_tokens: int = 0,
                    cost_eur: float = 0.0,
                    success: bool = True,
                    error_type: str = None,
        ):
                    """Record a chat task for dashboard metrics."""
                    self._ensure_initialized()
                    total_tokens = input_tokens + output_tokens
                    status = "success" if success else "failed"
                    cost_usd = cost_eur / 0.92 if cost_eur else 0.0

            if self._metrics:
                            self._metrics["tasks_total"].add(1, {"task_type": "chat", "status": status})
                            self._metrics["task_tokens"].record(total_tokens, {"task_type": "chat"})
                            if cost_usd > 0:
                                                self._metrics["task_cost"].record(cost_usd, {"task_type": "chat"})
                                            self._metrics["provider_requests"].add(1, {"provider": "routellm", "model": model, "success": str(success).lower()})
                            if not success and error_type:
                                                self._metrics["tool_errors"].add(1, {"tool": "chat", "error_type": error_type})
                                            savings = max(0, self._estimate_competitor_cost(total_tokens) - cost_usd)
                            if savings > 0:
                                                self._metrics["cost_savings"].add(savings)

            if success:
                            logger.info(f"Chat task: model={model}, tokens={total_tokens}, cost=EUR{cost_eur:.6f}")
            else:
                            logger.warning(f"Chat task failed: model={model}, error={error_type}")
                
    def _estimate_competitor_cost(self, total_tokens: int) -> float:
        avg_rate = sum(COMPETITOR_COSTS.values()) / len(COMPETITOR_COSTS)
        return (total_tokens / 1000) * avg_rate


def track_agent_task(task_type: str = "generic"):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            task_id = kwargs.get("task_id") or str(uuid.uuid4())[:8]
            with agent_telemetry.track_task(task_id, task_type) as ctx:
                if "task_ctx" in func.__code__.co_varnames:
                    kwargs["task_ctx"] = ctx
                return await func(*args, **kwargs)
        return wrapper
    return decorator


def track_tool_execution(tool_name: str):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            task_ctx = kwargs.get("task_ctx")
            with agent_telemetry.track_tool(tool_name, task_ctx):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


agent_telemetry = AgentTelemetry()


def generate_task_id() -> str:
    return str(uuid.uuid4())[:8]
