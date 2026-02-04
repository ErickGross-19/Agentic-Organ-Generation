"""
ScaffoldAgent - LLM-powered scaffold design assistant using multi-provider support.
"""

import json
import time
from typing import Any, Optional

from .prompts import SYSTEM_PROMPT
from .tools import (
    DEFAULT_SUGGESTIONS,
    GENERAL_SUGGESTIONS,
    SCAFFOLD_TOOLS,
    TOOL_TO_SCAFFOLD_TYPE,
)
from .tools_minimal import (
    MINIMAL_SCAFFOLD_TOOLS,
    MINIMAL_TOOL_TO_SCAFFOLD_TYPE,
)
from .providers import (
    BaseLLMClient,
    LLMClientFactory,
    LLMMessage,
    LLMProvider,
    LLMResponse,
)
from ..config import get_settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class ScaffoldAgent:
    """
    Multi-provider LLM scaffold design agent using function calling.

    This agent processes natural language requests and returns structured
    scaffold generation parameters. Supports multiple LLM providers.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the scaffold agent.

        Args:
            provider: LLM provider ("anthropic" or "openai"). If not provided, reads from settings.
            api_key: API key for the provider. If not provided, reads from settings.
            model: Model name. If not provided, uses provider default.
        """
        settings = get_settings()

        # Determine provider
        self.provider_name = provider or settings.llm_provider
        try:
            self.provider = LLMProvider(self.provider_name)
        except ValueError:
            logger.warning(f"Invalid provider '{self.provider_name}', defaulting to Anthropic")
            self.provider = LLMProvider.ANTHROPIC

        # Get API key
        if api_key:
            self.api_key = api_key
        elif self.provider == LLMProvider.ANTHROPIC:
            self.api_key = settings.anthropic_api_key
        elif self.provider == LLMProvider.OPENAI:
            self.api_key = settings.openai_api_key
        else:
            self.api_key = ""

        # Get model
        self.model = model or settings.llm_model

        # Get base URL for OpenAI-compatible endpoints
        self.base_url = settings.openai_base_url if self.provider == LLMProvider.OPENAI else None

        # Use minimal tools for OpenAI endpoints with limited context
        self.use_minimal_tools = self.provider == LLMProvider.OPENAI and bool(self.base_url)

        # Initialize client
        self.client: Optional[BaseLLMClient] = None
        if self.api_key:
            try:
                self.client = LLMClientFactory.create(
                    provider=self.provider,
                    api_key=self.api_key,
                    model=self.model if self.model else None,
                    base_url=self.base_url if self.base_url else None,
                )
                logger.info(f"Initialized {self.provider_name} client with model {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize {self.provider_name} client: {e}")

    @property
    def is_available(self) -> bool:
        """Check if the LLM client is available."""
        return self.client is not None

    async def chat(
        self,
        message: str,
        conversation_history: list[dict[str, str]],
        current_params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Process a chat message and return response with optional tool calls.

        Args:
            message: The user's message
            conversation_history: List of previous messages [{role, content}, ...]
            current_params: Current scaffold parameters for context

        Returns:
            {
                "message": str,  # Text response
                "action": "chat" | "generate" | "clarify",
                "params": dict | None,  # If action is generate
                "suggestions": list[str]  # Follow-up suggestions
            }
        """
        if not self.client:
            return self._fallback_response(message)

        try:
            # Build messages from history
            messages: list[LLMMessage] = []
            for msg in conversation_history[-20:]:  # Keep last 20 messages
                messages.append(LLMMessage(role=msg["role"], content=msg["content"]))
            messages.append(LLMMessage(role="user", content=message))

            # Add current params context to system prompt (shortened for minimal mode)
            if self.use_minimal_tools:
                system = "You are a bioprinting scaffold design assistant. Help users create scaffolds by calling the appropriate tool. Available types: vascular network, porous disc, lattice, tubular conduit, gyroid."
                if current_params:
                    system += f"\nCurrent: {current_params.get('type', 'none')}"
                tools = MINIMAL_SCAFFOLD_TOOLS
            else:
                system = SYSTEM_PROMPT
                if current_params:
                    system += f"\n\nCurrent scaffold parameters: {json.dumps(current_params, indent=2)}"
                tools = SCAFFOLD_TOOLS

            # Call LLM with tools
            response: LLMResponse = await self.client.chat_async(
                messages=messages,
                system=system,
                tools=tools,
                max_tokens=512 if self.use_minimal_tools else 1024,
            )

            logger.debug(f"LLM response received with {len(response.tool_calls)} tool calls")
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"{self.provider_name} API error: {e}")
            return self._error_response(str(e))

    def _parse_response(self, response: LLMResponse) -> dict[str, Any]:
        """
        Parse LLM response into structured format.

        Args:
            response: The LLMResponse from the provider

        Returns:
            Structured response dict
        """
        result: dict[str, Any] = {
            "message": response.text,
            "action": "chat",
            "params": None,
            "suggestions": [],
        }

        # Process tool calls (check both full and minimal tool mappings)
        tool_mapping = {**TOOL_TO_SCAFFOLD_TYPE, **MINIMAL_TOOL_TO_SCAFFOLD_TYPE}
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["input"]

            if tool_name in tool_mapping:
                # This is a scaffold generation tool
                scaffold_type = tool_mapping[tool_name]
                result["action"] = "generate"
                result["params"] = {
                    "type": scaffold_type,
                    **tool_input,
                }
                # Add seed if not provided
                if "seed" not in result["params"]:
                    result["params"]["seed"] = int(time.time()) % 10000

            elif tool_name == "ask_clarification":
                result["action"] = "clarify"
                result["message"] = tool_input.get("question", "")
                result["suggestions"] = tool_input.get("options", [])

        # Generate default suggestions if none provided
        if not result["suggestions"]:
            result["suggestions"] = self._generate_suggestions(
                result["action"], result["params"]
            )

        return result

    def _generate_suggestions(
        self, action: str, params: Optional[dict[str, Any]]
    ) -> list[str]:
        """
        Generate contextual follow-up suggestions.

        Args:
            action: The action type ("chat", "generate", "clarify")
            params: Scaffold parameters if action is "generate"

        Returns:
            List of suggestion strings
        """
        if action == "generate" and params:
            scaffold_type = params.get("type", "")
            if scaffold_type in DEFAULT_SUGGESTIONS:
                return DEFAULT_SUGGESTIONS[scaffold_type]

        return GENERAL_SUGGESTIONS

    def _fallback_response(self, message: str) -> dict[str, Any]:
        """
        Generate a fallback response when LLM is unavailable.

        Args:
            message: The user's message

        Returns:
            Fallback response with helpful guidance
        """
        message_lower = message.lower()

        # Simple keyword matching for basic functionality
        if any(word in message_lower for word in ["blood", "vessel", "vascular", "perfusion"]):
            return {
                "message": "I'll create a vascular network scaffold for you. (Note: Running without AI assistance)",
                "action": "generate",
                "params": {
                    "type": "vascular_network",
                    "inlets": 4,
                    "levels": 3,
                    "curvature": 0.5,
                    "deterministic": False,
                    "seed": int(time.time()) % 10000,
                },
                "suggestions": DEFAULT_SUGGESTIONS["vascular_network"],
            }
        elif any(word in message_lower for word in ["disc", "pore", "porous", "cell culture"]):
            return {
                "message": "I'll create a porous disc scaffold for you. (Note: Running without AI assistance)",
                "action": "generate",
                "params": {
                    "type": "porous_disc",
                    "diameter_mm": 10.0,
                    "height_mm": 2.0,
                    "pore_diameter_um": 200.0,
                    "pore_spacing_um": 400.0,
                },
                "suggestions": DEFAULT_SUGGESTIONS["porous_disc"],
            }
        elif any(word in message_lower for word in ["tube", "conduit", "nerve", "graft"]):
            return {
                "message": "I'll create a tubular conduit scaffold for you. (Note: Running without AI assistance)",
                "action": "generate",
                "params": {
                    "type": "tubular_conduit",
                    "outer_diameter_mm": 6.0,
                    "wall_thickness_mm": 1.0,
                    "length_mm": 20.0,
                    "inner_texture": "smooth",
                },
                "suggestions": DEFAULT_SUGGESTIONS["tubular_conduit"],
            }
        elif any(word in message_lower for word in ["lattice", "bone", "load", "mechanical"]):
            return {
                "message": "I'll create a lattice scaffold for you. (Note: Running without AI assistance)",
                "action": "generate",
                "params": {
                    "type": "lattice",
                    "bounding_box": {"x": 10, "y": 10, "z": 10},
                    "unit_cell": "cubic",
                    "cell_size_mm": 2.0,
                    "strut_diameter_mm": 0.5,
                },
                "suggestions": DEFAULT_SUGGESTIONS["lattice"],
            }
        else:
            return {
                "message": (
                    "I'm running without AI assistance. Please specify what type of scaffold you'd like:\n"
                    "- **Vascular network** for blood vessel scaffolds\n"
                    "- **Porous disc** for cell culture experiments\n"
                    "- **Tubular conduit** for nerve or vascular grafts\n"
                    "- **Lattice** for bone scaffolds\n"
                    "- **Primitive** for basic shapes"
                ),
                "action": "chat",
                "params": None,
                "suggestions": GENERAL_SUGGESTIONS,
            }

    def _error_response(self, error: str) -> dict[str, Any]:
        """
        Generate an error response.

        Args:
            error: Error message

        Returns:
            Error response dict
        """
        return {
            "message": f"I encountered an error: {error}. Please try again or rephrase your request.",
            "action": "chat",
            "params": None,
            "suggestions": ["Try again", "Create a simple scaffold", "Help"],
        }


# Singleton instance for reuse
_agent_instance: Optional[ScaffoldAgent] = None


def get_agent() -> ScaffoldAgent:
    """
    Get or create the singleton ScaffoldAgent instance.

    Returns:
        ScaffoldAgent instance
    """
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ScaffoldAgent()
    return _agent_instance
