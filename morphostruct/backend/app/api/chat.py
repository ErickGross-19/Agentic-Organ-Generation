"""
Chat/LLM REST API endpoints.

Provides endpoints for:
- POST /chat: Send message to LLM assistant using ScaffoldAgent

Integrates with the ScaffoldAgent for intelligent scaffold design assistance.
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime

from ..llm.agent import ScaffoldAgent, get_agent as create_agent
from ..core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])

# Global agent instance to maintain conversation state across requests
_agent_instance = None


def get_agent() -> ScaffoldAgent:
    """Get or create the global ScaffoldAgent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = create_agent()
    return _agent_instance


# ============================================================================
# Request/Response Models
# ============================================================================


class ChatMessageHistory(BaseModel):
    """A message in conversation history."""

    role: str = Field(description="Message role: 'user' or 'assistant'")
    content: str = Field(description="Message content")


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    message: str = Field(description="User's message")
    conversation_history: List[ChatMessageHistory] = Field(
        default_factory=list,
        description="Previous conversation messages for context",
    )
    current_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Current scaffold parameters for context",
    )


class ChatResponse(BaseModel):
    """Response from chat endpoint."""

    message: str = Field(description="Assistant's response message")
    action: Optional[str] = Field(
        default=None,
        description="Suggested action: 'generate', 'modify', 'explain', or None",
    )
    suggested_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Suggested parameter changes (if action is 'generate' or 'modify')",
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Clickable suggestion chips for follow-up",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="Response timestamp (ISO format)",
    )


# ============================================================================
# Endpoints
# ============================================================================


def _generate_placeholder_response(
    message: str,
    current_params: Optional[Dict[str, Any]] = None,
) -> ChatResponse:
    """
    Generate a placeholder response when LLM is unavailable.

    Args:
        message: The user's message
        current_params: Current scaffold parameters

    Returns:
        ChatResponse with helpful guidance
    """
    return ChatResponse(
        message=(
            "I'm running without AI assistance. Please specify what type of scaffold you'd like:\n"
            "- **Vascular network** for blood vessel scaffolds\n"
            "- **Porous disc** for cell culture experiments\n"
            "- **Tubular conduit** for nerve or vascular grafts\n"
            "- **Lattice** for bone scaffolds"
        ),
        action=None,
        suggested_params=None,
        suggestions=[
            "Create a vascular network",
            "Create a porous disc",
            "Create a tubular conduit",
            "Create a lattice",
        ],
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the LLM assistant.

    Uses ScaffoldAgent for intelligent scaffold design assistance. The agent can:
    - Understand natural language requests for scaffold creation
    - Suggest appropriate parameters based on the request
    - Explain scaffold types and parameters
    - Recommend modifications to existing designs
    - Work in LLM mode (with API key) or fallback mode (keyword-based)
    """
    try:
        agent = get_agent()
        logger.info(f"Processing chat message: {request.message[:100]}...")

        if agent.is_available:
            # Convert history format
            history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]

            result = await agent.chat(
                message=request.message,
                conversation_history=history,
                current_params=request.current_params,
            )

            logger.info(f"Chat response generated: action={result.get('action')}")
            return ChatResponse(
                message=result.get("message", ""),
                action=result.get("action") if result.get("action") not in ("chat", "clarify") else None,
                suggested_params=result.get("params"),
                suggestions=result.get("suggestions", []),
            )
        else:
            # Fallback to placeholder
            logger.warning("LLM not available, using placeholder response")
            return _generate_placeholder_response(
                request.message,
                request.current_params,
            )
    except Exception as e:
        logger.exception(f"Chat processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}",
        )


@router.get("/chat/status")
async def chat_status() -> Dict[str, Any]:
    """
    Get chat service status.

    Returns information about the chat backend configuration.
    """
    agent = get_agent()
    return {
        "status": "active" if agent.is_available else "placeholder",
        "llm_configured": agent.is_available,
        "message": "LLM assistant ready" if agent.is_available else "Running in placeholder mode",
    }


