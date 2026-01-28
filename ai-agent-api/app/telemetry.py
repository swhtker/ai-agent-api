"""
Punky API Telemetry Module
Sends traces, metrics, and logs to the Helsinki monitoring stack.
"""

import logging
import os
from contextlib import asynccontextmanager
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


# Configuration - Update these for your environment
OTEL_ENDPOINT = os.getenv("OTEL_ENDPOINT", "89.167.6.124:4317")
OTEL_ENABLED = os.getenv("OTEL_ENABLED", "true").lower() == "true"
SERVICE_NAME_VALUE = os.getenv("SERVICE_NAME", "punky-api")
SERVICE_VERSION_VALUE = os.getenv("SERVICE_VERSION", "1.0.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")


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
        insecure=True,  # Set to False if using TLS
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
        export_interval_millis=30000,  # Export every 30 seconds
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
    
    # Add OTLP handler to root logger
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
        if self._initialized:
            return
        
        if not OTEL_ENABLED:
            logging.info("OpenTelemetry disabled via OTEL_ENABLED=false")
            self._initialized = True
            return
        
        try:
            resource = create_resource()
            
            self.tracer_provider = setup_tracing(resource)
            self.meter_provider = setup_metrics(resource)
            self.logger_provider = setup_logging(resource)
            
            # Initialize custom metrics
            self._metrics = self._create_custom_metrics()
            
            self._initialized = True
            logging.info(f"Telemetry initialized - sending to {OTEL_ENDPOINT}")
        except Exception as e:
            logging.error(f"Failed to initialize telemetry: {e}")
            self._initialized = True  # Mark as initialized to prevent retries
    
    def _create_custom_metrics(self):
        """Create custom metrics for business logic monitoring."""
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
            "llm_input_tokens": meter.create_counter(
                name="punky_llm_input_tokens_total",
                description="Total LLM input tokens consumed",
                unit="1",
            ),
            "llm_output_tokens": meter.create_counter(
                name="punky_llm_output_tokens_total",
                description="Total LLM output tokens consumed",
                unit="1",
            ),
            "llm_tokens_total": meter.create_counter(
                name="punky_llm_tokens_total",
                description="Total LLM tokens consumed (input + output)",
                unit="1",
            ),
            "browser_sessions": meter.create_up_down_counter(
                name="punky_browser_sessions",
                description="Number of active browser sessions",
                unit="1",
            ),
            "llm_requests": meter.create_counter(
                name="punky_llm_requests_total",
                description="Total number of LLM requests",
                unit="1",
            ),
        }
    
    @property
    def metrics(self):
        """Get the metrics dictionary."""
        return self._metrics
    
    def instrument_fastapi(self, app):
        """Instrument FastAPI application."""
        if OTEL_ENABLED:
            FastAPIInstrumentor.instrument_app(app)
    
    def instrument_httpx(self):
        """Instrument HTTPX client for outgoing requests."""
        if OTEL_ENABLED:
            HTTPXClientInstrumentor().instrument()
    
    def instrument_sqlalchemy(self, engine):
        """Instrument SQLAlchemy for database tracing."""
        if OTEL_ENABLED:
            SQLAlchemyInstrumentor().instrument(engine=engine)
    
    async def shutdown(self):
        """Gracefully shutdown telemetry exporters."""
        if self.tracer_provider:
            self.tracer_provider.shutdown()
        if self.meter_provider:
            self.meter_provider.shutdown()
        if self.logger_provider:
            self.logger_provider.shutdown()
        
        logging.info("Telemetry shutdown complete")


# Global telemetry manager instance
telemetry = TelemetryManager()


# ============================================================
# Helper functions for recording metrics - THESE WERE MISSING!
# ============================================================

def record_llm_tokens(input_tokens: int, output_tokens: int, model: str = "unknown"):
    """Record LLM token usage metrics."""
    if not OTEL_ENABLED or telemetry.metrics is None:
        return
    
    try:
        attributes = {"model": model}
        
        telemetry.metrics["llm_input_tokens"].add(input_tokens, attributes)
        telemetry.metrics["llm_output_tokens"].add(output_tokens, attributes)
        telemetry.metrics["llm_tokens_total"].add(input_tokens + output_tokens, attributes)
        telemetry.metrics["llm_requests"].add(1, attributes)
    except Exception as e:
        logging.warning(f"Failed to record LLM tokens: {e}")


def record_browser_session_start():
    """Record when a browser session starts."""
    if not OTEL_ENABLED or telemetry.metrics is None:
        return
    
    try:
        telemetry.metrics["browser_sessions"].add(1)
    except Exception as e:
        logging.warning(f"Failed to record browser session start: {e}")


def record_browser_session_end():
    """Record when a browser session ends."""
    if not OTEL_ENABLED or telemetry.metrics is None:
        return
    
    try:
        telemetry.metrics["browser_sessions"].add(-1)
    except Exception as e:
        logging.warning(f"Failed to record browser session end: {e}")


def record_task_start():
    """Record when an agent task starts."""
    if not OTEL_ENABLED or telemetry.metrics is None:
        return
    
    try:
        telemetry.metrics["active_tasks"].add(1)
    except Exception as e:
        logging.warning(f"Failed to record task start: {e}")


def record_task_end(duration_seconds: float):
    """Record when an agent task ends."""
    if not OTEL_ENABLED or telemetry.metrics is None:
        return
    
    try:
        telemetry.metrics["active_tasks"].add(-1)
        telemetry.metrics["task_duration"].record(duration_seconds)
    except Exception as e:
        logging.warning(f"Failed to record task end: {e}")


def record_request():
    """Record an API request."""
    if not OTEL_ENABLED or telemetry.metrics is None:
        return
    
    try:
        telemetry.metrics["requests_total"].add(1)
    except Exception as e:
        logging.warning(f"Failed to record request: {e}")


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
