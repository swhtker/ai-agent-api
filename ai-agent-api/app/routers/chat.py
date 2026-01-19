from datetime import datetime
from typing import List
import logging

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app import models

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    id: str
    role: str
    content: str
    timestamp: str


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    Simple ChatGPT-like chat endpoint:
    - takes a list of messages [{role, content}]
    - calls your model
    - returns assistant reply
    """
    try:
        # Get last user message
        user_message = next(
            (m for m in reversed(request.messages) if m.role == "user"),
            None,
        )
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message provided")

        # TODO: replace this with your real model call
        # e.g. call OpenAI, Claude, local model, or your worker pipeline
        assistant_text = (
            "This is a placeholder answer. "
            "Wire this function to your real AI worker or model."
        )

        reply_id = f"chat_{int(datetime.utcnow().timestamp() * 1000)}"
        timestamp = datetime.utcnow().isoformat()

        # Optionally store chat in Task table
        user_task = models.Task(
            name=f"Chat user: {user_message.content[:50]}",
            description=user_message.content,
            task_type="chat",
            status="completed",
            metainfo={
                "role": "user",
                "timestamp": timestamp,
                "message_id": reply_id + "_user",
            },
        )

        assistant_task = models.Task(
            name=f"Chat reply to: {reply_id}",
            description=assistant_text,
            task_type="chat",
            status="completed",
            metainfo={
                "role": "assistant",
                "timestamp": timestamp,
                "message_id": reply_id,
            },
        )

        db.add(user_task)
        db.add(assistant_task)
        db.commit()

        return ChatResponse(
            id=reply_id,
            role="assistant",
            content=assistant_text,
            timestamp=timestamp,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
