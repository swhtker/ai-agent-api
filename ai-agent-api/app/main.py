from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .config import get_settings
from .database import get_db, init_db, engine
from . import models
from .routers import training, chat
from .telemetry import telemetry, add_span_attributes, record_exception

# NEW: Import production metrics
from .prometheus_metrics_production import (
    initialize as init_metrics,
    shutdown as shutdown_metrics,
    create_metrics_router,
    PrometheusMiddleware,
    pricing_discovery
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize and shutdown resources."""
    # Startup
    init_db()

    # Initialize telemetry
    telemetry.initialize()
    telemetry.instrument_fastapi(app)
    telemetry.instrument_httpx()
    telemetry.instrument_sqlalchemy(engine)

    # NEW: Initialize Prometheus metrics with auto-discovery
    # This starts background scheduler for daily pricing refresh
    await init_metrics()

    yield

    # Shutdown
    # NEW: Graceful metrics shutdown (saves cache, stops scheduler)
    await shutdown_metrics()
    
    await telemetry.shutdown()


# Initialize FastAPI with lifespan
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

# CORS
raw_origins = settings.CORS_ORIGINS or "https://jspriggins.com"
if raw_origins == "*":
    origins = ["*"]
else:
    origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NEW: Add Prometheus metrics middleware
# This automatically tracks all HTTP requests (latency, status codes)
app.add_middleware(PrometheusMiddleware)

# Include routers
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])

# NEW: Add metrics management routes
# Provides: /metrics, /metrics/health, /metrics/discovery, /metrics/refresh, etc.
app.include_router(create_metrics_router())

# API Key Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(api_key: str = Security(api_key_header)):
    if not settings.API_KEY:
        return None
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return api_key


# Request/Response Models
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    metrics_status: Optional[Dict[str, Any]] = None  # NEW: Include metrics health


class TaskCreate(BaseModel):
    name: str = Field(..., description="Task name")
    description: Optional[str] = Field(None, description="Task description")
    task_type: str = Field("general", description="Type of task")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class TaskResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    task_type: str
    status: str
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


# Endpoints
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint for monitoring."""
    # NEW: Include metrics system health
    metrics_health = pricing_discovery.get_health()
    
    return HealthResponse(
        status="healthy" if metrics_health["status"] == "healthy" else "degraded",
        timestamp=datetime.utcnow(),
        version=settings.APP_VERSION,
        metrics_status=metrics_health,
    )


@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics",
            "metrics_health": "/metrics/health",
            "metrics_discovery": "/metrics/discovery",
        }
    }


@app.get("/api/tasks", response_model=list[TaskResponse], tags=["tasks"])
async def list_tasks(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key),
):
    """List all tasks."""
    add_span_attributes({"task.operation": "list", "task.skip": skip, "task.limit": limit})
    tasks = db.query(models.Task).offset(skip).limit(limit).all()
    return tasks


@app.post("/api/tasks", response_model=TaskResponse, status_code=status.HTTP_201_CREATED, tags=["tasks"])
async def create_task(
    task: TaskCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key),
):
    """Create a new task."""
    add_span_attributes({
        "task.operation": "create",
        "task.name": task.name,
        "task.type": task.task_type,
    })

    try:
        db_task = models.Task(
            name=task.name,
            description=task.description,
            task_type=task.task_type,
            parameters=task.parameters,
            status="pending",
        )
        db.add(db_task)
        db.commit()
        db.refresh(db_task)
        return db_task
    except Exception as e:
        record_exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tasks/{task_id}", response_model=TaskResponse, tags=["tasks"])
async def get_task(
    task_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key),
):
    """Get a specific task by ID."""
    add_span_attributes({"task.operation": "get", "task.id": task_id})

    task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.delete("/api/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["tasks"])
async def delete_task(
    task_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key),
):
    """Delete a task."""
    add_span_attributes({"task.operation": "delete", "task.id": task_id})

    task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    db.delete(task)
    db.commit()
    return None
