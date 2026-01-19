"""
Training Router - Handles AI Training Chat from WordPress

Provides endpoints for:
- POST /api/training/message - Send training instruction and get response
- GET /api/training/history - Retrieve training conversation history
"""
from datetime import datetime
from typing import List, Optional
import logging

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app import models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["training"])


# Pydantic models for request/response
class TrainingMessageRequest(BaseModel):
    """Request model for training message"""
    message: str
    context: Optional[dict] = None


class TrainingMessageResponse(BaseModel):
    """Response model for training message"""
    id: str
    role: str
    content: str
    timestamp: str


class TrainingHistoryItem(BaseModel):
    """History item model"""
    id: str
    role: str
    content: str
    timestamp: str


class TrainingHistoryResponse(BaseModel):
    """Response model for training history"""
    messages: List[TrainingHistoryItem]
    total: int


@router.post("/message", response_model=TrainingMessageResponse)
async def send_training_message(
    request: TrainingMessageRequest,
    db: Session = Depends(get_db)
):
    """
    Send a training message and receive a response.
    """
    try:
        message_id = f"msg_{int(datetime.utcnow().timestamp() * 1000)}"
        timestamp = datetime.utcnow().isoformat()

        # Store training in Task model
        user_task = models.Task(
            title=f"Training: {request.message[:50]}",
            description=request.message,
            task_type="training",
            status="completed",
            input_data={"role": "user", "context": request.context},
            metadata_={"timestamp": timestamp, "message_id": message_id}
        )
        db.add(user_task)

        response_content = f"Training instruction received: '{request.message[:100]}{'...' if len(request.message) > 100 else ''}'. Stored for Punky's learning!"

        response_id = f"msg_{int(datetime.utcnow().timestamp() * 1000) + 1}"
        response_timestamp = datetime.utcnow().isoformat()

        assistant_task = models.Task(
            title=f"Response to: {message_id}",
            description=response_content,
            task_type="training",
            status="completed",
            input_data={"role": "assistant", "in_response_to": message_id},
            metadata_={"timestamp": response_timestamp, "message_id": response_id}
        )
        db.add(assistant_task)
        db.commit()

        logger.info(f"Training message stored: {message_id}")

        return TrainingMessageResponse(
            id=response_id,
            role="assistant",
            content=response_content,
            timestamp=response_timestamp
        )

    except Exception as e:
        logger.error(f"Error processing training message: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


@router.get("/history", response_model=TrainingHistoryResponse)
async def get_training_history(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get training conversation history."""
    try:
        tasks = db.query(models.Task).filter(
            models.Task.task_type == "training"
        ).order_by(models.Task.created_at.asc()).offset(offset).limit(limit).all()

        total = db.query(models.Task).filter(
            models.Task.task_type == "training"
        ).count()

        messages = []
        for task in tasks:
            input_data = task.input_data or {}
            metadata = task.metadata_ or {}
            role = input_data.get("role", "user")
            timestamp = metadata.get("timestamp", task.created_at.isoformat())

            messages.append(TrainingHistoryItem(
                id=metadata.get("message_id", str(task.id)),
                role=role,
                content=task.description or "",
                timestamp=timestamp
            ))

        return TrainingHistoryResponse(messages=messages, total=total)

    except Exception as e:
        logger.error(f"Error retrieving training history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")
