from datetime import datetime
from typing import List
import logging
import httpx
import os

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app import models

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

# RouteLLM router configuration
ROUTER_URL = os.getenv("ROUTER_URL", "http://routellm-router:4000")
ROUTER_MODEL = "router-mf-0.11593"  # RouteLLM model with threshold


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


async def call_router(messages: List[dict]) -> str:
    """
    Call RouteLLM router which automatically routes between:
    - Strong model: Claude 3.5 Sonnet (complex queries)
    - Weak model: Groq Llama 3.1 8B (simple queries)

    The MF (Matrix Factorization) classifier decides which model to use.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{ROUTER_URL}/v1/chat/completions",
                json={
                    "model": ROUTER_MODEL,
                    "messages": messages,
                    "temperature": 0.7,
                },
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logger.error(f"Router HTTP error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=502, detail=f"Router error: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Router connection error: {str(e)}")
            raise HTTPException(status_code=503, detail="Could not connect to AI router")
        except Exception as e:
            logger.error(f"Unexpected error calling router: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    Simple ChatGPT-like chat endpoint:
    - takes a list of messages [{role, content}]
    - calls RouteLLM router (auto-routes to Claude or Groq)
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

        # Convert to dict format for API call
        messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]

        # Call RouteLLM router - it automatically decides Claude vs Groq
        assistant_text = await call_router(messages_dict)

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
                "message_id": reply_id + "_assistant",
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
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
