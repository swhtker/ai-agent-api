"""
Punky API Telemetry Module
Sends traces, metrics, and logs to the Helsinki monitoring stack.
"""

import logging
import os
from typing import Optional

from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter


# Configuration from environment
OTEL_ENDPOINT = os.getenv("OTEL_ENDPOINT", "89.167.6.124:4317")
OTEL_ENABLED = os.getenv("OTEL_ENABLED", "true").lower() == "true"
SERVICE_NAME_VALUE = os.getenv("SERVICE_NAME", "punky-api")
SERVICE_VERSION_VALUE = os.getenv("SERVICE_VERSION", "1.0.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

logger = logging.getLogger(__name__)


def create_resource() -> Resource:
    """Create OpenTelemetry resource with service metadata."""
    return Resource.create({
        SERVICE_NAME: SERVICE_NAME_VALUE,
        SERVICE_VERSION: SERVICE_VERSION_VALUE,
        "deployment.environment": ENVIRONMENT,
        "service.namespace": "punky",
    })


def setup_tracing(resource: Resource) -> TracerProvider:
    """Configure distributed tracing with Tempo."""
    provider = TracerProvider(resource=resource)
    
    exporter = OTLPSpanExporter(
        endpoint=OTEL_ENDPOINT,
        insecure=True,
    )
    
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    
    trace.set_tracer_provider(provider)
    return provider


def setup_metrics(resource: Resource) -> MeterProvider:
    """Configure metrics export to VictoriaMetrics via OTLP."""
    exporter = OTLPMetricExporter(
        endpoint=OTEL_ENDPOINT,
        insecure=True,
    )
    
    reader = PeriodicExportingMetricReader(
        exporter,
        export_interval_millis=30000,
    )
    
    provider = MeterProvider(
        resource=resource,
        metric_readers=[reader],
    )
    
    metrics.set_meter_provider(provider)
    return provider


def setup_logging(resource: Resource) -> LoggerProvider:
    """Configure log export to Loki via OTLP."""
    provider = LoggerProvider(resource=resource)
    
    exporter = OTLPLogExporter(
        endpoint=OTEL_ENDPOINT,
        insecure=True,
    )
    
    processor = BatchLogRecordProcessor(exporter)
    provider.add_log_record_processor(processor)
    
    set_logger_provider(provider)
    
    handler = LoggingHandler(
        level=logging.INFO,
        logger_provider=provider,
    )
    logging.getLogger().addHandler(handler)
    
    return provider


class TelemetryManager:
    """Manages OpenTelemetry lifecycle."""
    
    def __init__(self):
        self.tracer_provider: Optional[TracerProvider] = None
        self.meter_provider: Optional[MeterProvider] = None
        self.logger_provider: Optional[LoggerProvider] = None
        self._initialized = False
        self._metrics = None
    
    def initialize(self):
        """Initialize all telemetry components."""
        if self._initialized or not OTEL_ENABLED:
            if not OTEL_ENABLED:
                logger.info("Telemetry disabled via OTEL_ENABLED=false")
            return
        
        try:
            resource = create_resource()
            
            self.tracer_provider = setup_tracing(resource)
            self.meter_provider = setup_metrics(resource)
            self.logger_provider = setup_logging(resource)
            
            self._initialized = True
            logger.info(f"Telemetry initialized - sending to {OTEL_ENDPOINT}")
        except Exception as e:
            logger.error(f"Failed to initialize telemetry: {e}")
    
    def instrument_fastapi(self, app):
        """Instrument FastAPI application."""
        if not OTEL_ENABLED:
            return
        try:
            FastAPIInstrumentor.instrument_app(app)
        except Exception as e:
            logger.error(f"Failed to instrument FastAPI: {e}")
    
    def instrument_httpx(self):
        """Instrument HTTPX client for outgoing requests."""
        if not OTEL_ENABLED:
            return
        try:
            HTTPXClientInstrumentor().instrument()
        except Exception as e:
            logger.error(f"Failed to instrument HTTPX: {e}")
    
    def instrument_sqlalchemy(self, engine):
        """Instrument SQLAlchemy for database tracing."""
        if not OTEL_ENABLED:
            return
        try:
            SQLAlchemyInstrumentor().instrument(engine=engine)
        except Exception as e:
            logger.error(f"Failed to instrument SQLAlchemy: {e}")
    
    def get_metrics(self):
        """Get or create custom metrics."""
        if self._metrics is None and OTEL_ENABLED:
            self._metrics = create_custom_metrics()
        return self._metrics
    
    async def shutdown(self):
        """Gracefully shutdown telemetry exporters."""
        if self.tracer_provider:
            self.tracer_provider.shutdown()
        if self.meter_provider:
            self.meter_provider.shutdown()
        if self.logger_provider:
            self.logger_provider.shutdown()
        
        logger.info("Telemetry shutdown complete")


# Global telemetry manager instance
telemetry = TelemetryManager()


def create_custom_metrics():
    """Create custom metrics for Punky API monitoring."""
    if not OTEL_ENABLED:
        return None
    
    meter = metrics.get_meter(SERVICE_NAME_VALUE)
    
    return {
        "requests_total": meter.create_counter(
            name="punky_requests_total",
            description="Total number of API requests",
            unit="1",
        ),
        "active_tasks": meter.create_up_down_counter(
            name="punky_active_tasks",
            description="Number of currently active agent tasks",
            unit="1",
        ),
        "task_duration": meter.create_histogram(
            name="punky_task_duration_seconds",
            description="Duration of agent tasks",
            unit="s",
        ),
        "llm_tokens_used": meter.create_counter(
            name="punky_llm_tokens_total",
            description="Total LLM tokens consumed",
            unit="1",
        ),
        "llm_requests": meter.create_counter(
            name="punky_llm_requests_total",
            description="Total LLM API requests",
            unit="1",
        ),
        "browser_sessions": meter.create_up_down_counter(
            name="punky_browser_sessions",
            description="Number of active browser sessions",
            unit="1",
        ),
        "training_jobs": meter.create_counter(
            name="punky_training_jobs_total",
            description="Total training jobs processed",
            unit="1",
        ),
    }


# Tracing helpers
def get_tracer():
    """Get a tracer for manual span creation."""
    return trace.get_tracer(SERVICE_NAME_VALUE)


def add_span_attributes(attributes: dict):
    """Add attributes to the current span."""
    span = trace.get_current_span()
    if span:
        for key, value in attributes.items():
            span.set_attribute(key, value)


def record_exception(exception: Exception):
    """Record an exception in the current span."""
    span = trace.get_current_span()
    if span:
        span.record_exception(exception)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))


# Metric recording helpers
def record_llm_tokens(input_tokens: int, output_tokens: int, model: str = "unknown"):
    """Record LLM token usage."""
    metrics_obj = telemetry.get_metrics()
    if metrics_obj:
        metrics_obj["llm_tokens_used"].add(
            input_tokens + output_tokens,
            attributes={"model": model, "token_type": "total"}
        )
        metrics_obj["llm_requests"].add(1, attributes={"model": model})


def record_task_start():
    """Record a task starting."""
    metrics_obj = telemetry.get_metrics()
    if metrics_obj:
        metrics_obj["active_tasks"].add(1)


def record_task_end(duration_seconds: float, status: str = "success"):
    """Record a task ending."""
    metrics_obj = telemetry.get_metrics()
    if metrics_obj:
        metrics_obj["active_tasks"].add(-1)
        metrics_obj["task_duration"].record(duration_seconds, attributes={"status": status})


def record_browser_session_start():
    """Record a browser session starting."""
    metrics_obj = telemetry.get_metrics()
    if metrics_obj:
        metrics_obj["browser_sessions"].add(1)


def record_browser_session_end():
    """Record a browser session ending."""
    metrics_obj = telemetry.get_metrics()
    if metrics_obj:
        metrics_obj["browser_sessions"].add(-1)


def record_training_job(job_type: str, status: str = "completed"):
    """Record a training job."""
    metrics_obj = telemetry.get_metrics()
    if metrics_obj:
        metrics_obj["training_jobs"].add(1, attributes={"job_type": job_type, "status": status})
