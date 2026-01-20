from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .config import get_settings
from .database import get_db, init_db
from . import models
from .routers import training, chat  # include chat router

settings = get_settings()

# Initialize FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
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

# Include routers
# Training router for WordPress AI Training Chat
app.include_router(training.router, prefix="/api")
# Chat router for interactive chat
app.include_router(chat.router, prefix="/api")

# API Key Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Simple API key verification"""
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key


# Pydantic schemas
class TaskCreate(BaseModel):
    title: str = Field(..., max_length=255)
    description: Optional[str] = None
    task_type: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class TaskUpdate(BaseModel):
    title: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    status: Optional[str] = None
    output_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()
    print(f"✅ {settings.APP_NAME} v{settings.APP_VERSION} started")


# Routes
@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "sqlite",
    }


@app.post("/tasks", status_code=status.HTTP_201_CREATED)
async def create_task(
    task: TaskCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """Create a new task"""
    db_task = models.Task(**task.model_dump())
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task.to_dict()


@app.get("/tasks/{task_id}")
async def get_task(
    task_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """Get a specific task by ID"""
    task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task.to_dict()


@app.get("/tasks")
async def list_tasks(
    status: Optional[str] = None,
    task_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """List tasks with optional filtering"""
    query = db.query(models.Task)

    if status:
        query = query.filter(models.Task.status == status)
    if task_type:
        query = query.filter(models.Task.task_type == task_type)

    tasks = (
        query.order_by(models.Task.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "tasks": [task.to_dict() for task in tasks],
        "count": len(tasks),
        "offset": offset,
        "limit": limit,
    }


@app.patch("/tasks/{task_id}")
async def update_task(
    task_id: int,
    task_update: TaskUpdate,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """Update a task"""
    task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    update_data = task_update.model_dump(exclude_unset=True)

    # Auto-set completed_at when status changes to completed
    if update_data.get("status") == "completed" and task.status != "completed":
        update_data["completed_at"] = datetime.utcnow()

    for key, value in update_data.items():
        setattr(task, key, value)

    db.commit()
    db.refresh(task)
    return task.to_dict()


@app.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """Delete a task"""
    task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    db.delete(task)
    db.commit()
    return None
