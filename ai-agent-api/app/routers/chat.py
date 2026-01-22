from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import httpx
import os
import logging

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

async def call_router(messages: List[dict]) -> dict:
    """Call the RouteLLM router to get a response."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            logger.info(f"Calling router at {ROUTER_URL}")
            logger.info(f"Using model: {ROUTER_MODEL}")
            logger.info(f"Messages count: {len(messages)}")

            response = await client.post(
                ROUTER_URL,
                json={
                    "model": ROUTER_MODEL,
                    "messages": messages,
                    "temperature": 0.7,
                    "drop_params": True,
                },
                headers={"Content-Type": "application/json"},
            )

            logger.info(f"Router response status: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"Router error response: {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Router error: {response.text}"
                )

            result = response.json()
            logger.info(f"Router response model: {result.get('model', 'unknown')}")
            return result

        except httpx.TimeoutException:
            logger.error("Router request timed out")
            raise HTTPException(status_code=504, detail="Router request timed out")
        except httpx.RequestError as e:
            logger.error(f"Router request failed: {str(e)}")
            raise HTTPException(status_code=503, detail=f"Router unavailable: {str(e)}")

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint that routes requests through RouteLLM.
    The router will automatically select the best model based on query complexity.
    """
    try:
        # Convert messages to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        logger.info(f"Processing chat request with {len(messages)} messages")

        # Call the router
        result = await call_router(messages)

        # Extract the response
        if "choices" in result and len(result["choices"]) > 0:
            response_content = result["choices"][0]["message"]["content"]
            model_used = result.get("model", "unknown")

            logger.info(f"Chat response generated using model: {model_used}")

            return ChatResponse(
                response=response_content,
                model_used=model_used
            )
        else:
            logger.error(f"Unexpected response format: {result}")
            raise HTTPException(status_code=500, detail="Unexpected response format from router")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "router_url": ROUTER_URL, "router_model": ROUTER_MODEL}
