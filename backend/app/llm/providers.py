"""
Multi-provider LLM abstraction layer.

Supports multiple LLM providers (Anthropic, OpenAI) with a unified interface.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

import anthropic


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class LLMMessage:
    """Provider-agnostic message representation."""

    def __init__(self, role: str, content: str):
        """
        Create a message.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
        """
        self.role = role
        self.content = content

    def to_anthropic(self) -> dict[str, str]:
        """Convert to Anthropic message format."""
        return {"role": self.role, "content": self.content}

    def to_openai(self) -> dict[str, str]:
        """Convert to OpenAI message format."""
        return {"role": self.role, "content": self.content}


class LLMResponse:
    """Provider-agnostic response representation."""

    def __init__(
        self,
        text: str,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        raw_response: Optional[Any] = None,
    ):
        """
        Create a response.

        Args:
            text: Response text content
            tool_calls: List of tool calls [{name, input}, ...]
            raw_response: Raw provider response for debugging
        """
        self.text = text
        self.tool_calls = tool_calls or []
        self.raw_response = raw_response


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        system: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
    ) -> LLMResponse:
        """
        Send a chat request to the LLM.

        Args:
            messages: List of messages in the conversation
            system: Optional system prompt
            tools: Optional tool definitions
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)

        Returns:
            LLMResponse with text and optional tool calls
        """
        pass

    @abstractmethod
    async def chat_async(
        self,
        messages: list[LLMMessage],
        system: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
    ) -> LLMResponse:
        """Async version of chat."""
        pass


class AnthropicClient(BaseLLMClient):
    """Anthropic (Claude) client implementation."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key
            model: Model name to use
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def chat(
        self,
        messages: list[LLMMessage],
        system: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
    ) -> LLMResponse:
        """Send sync chat request to Anthropic."""
        # Convert messages to Anthropic format
        anthropic_messages = [msg.to_anthropic() for msg in messages]

        # Build request parameters
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_messages,
        }

        if system:
            params["system"] = system

        if tools:
            params["tools"] = tools

        # Call API
        response = self.client.messages.create(**params)

        # Parse response
        return self._parse_response(response)

    async def chat_async(
        self,
        messages: list[LLMMessage],
        system: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
    ) -> LLMResponse:
        """Send async chat request to Anthropic."""
        # Anthropic SDK doesn't have native async support yet,
        # so we use the sync version for now
        return self.chat(messages, system, tools, max_tokens, temperature)

    def _parse_response(self, response: anthropic.types.Message) -> LLMResponse:
        """Parse Anthropic response into LLMResponse."""
        text = ""
        tool_calls: list[dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                tool_calls.append({"name": block.name, "input": block.input})

        return LLMResponse(text=text, tool_calls=tool_calls, raw_response=response)


class OpenAIClient(BaseLLMClient):
    """OpenAI client implementation."""

    def __init__(self, api_key: str, model: str = "gpt-4", base_url: Optional[str] = None):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model name to use
            base_url: Custom base URL for OpenAI-compatible endpoints
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI provider requires 'openai' package. "
                "Install with: pip install openai"
            )

        # Support custom OpenAI-compatible endpoints (e.g., Harvard proxy, Azure)
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = openai.OpenAI(**client_kwargs)
        self.base_url = base_url
        self.model = model

    def chat(
        self,
        messages: list[LLMMessage],
        system: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
    ) -> LLMResponse:
        """Send sync chat request to OpenAI."""
        # Convert messages to OpenAI format
        openai_messages = []

        if system:
            openai_messages.append({"role": "system", "content": system})

        openai_messages.extend([msg.to_openai() for msg in messages])

        # Build request parameters
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": openai_messages,
        }

        if tools:
            # Convert Anthropic tool format to OpenAI function format
            params["functions"] = self._convert_tools_to_functions(tools)

        # Call API
        response = self.client.chat.completions.create(**params)

        # Parse response
        return self._parse_response(response)

    async def chat_async(
        self,
        messages: list[LLMMessage],
        system: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
    ) -> LLMResponse:
        """Send async chat request to OpenAI."""
        # OpenAI SDK has async support
        import openai

        client_kwargs = {"api_key": self.client.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        async_client = openai.AsyncOpenAI(**client_kwargs)

        # Convert messages
        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend([msg.to_openai() for msg in messages])

        # Build request
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": openai_messages,
        }

        if tools:
            params["functions"] = self._convert_tools_to_functions(tools)

        # Call API
        response = await async_client.chat.completions.create(**params)

        # Parse response
        return self._parse_response(response)

    def _convert_tools_to_functions(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Convert Anthropic tool format to OpenAI function format.

        Args:
            tools: List of Anthropic-style tools

        Returns:
            List of OpenAI-style functions
        """
        functions = []
        for tool in tools:
            function = {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            }
            functions.append(function)
        return functions

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse OpenAI response into LLMResponse."""
        choice = response.choices[0]
        message = choice.message

        text = message.content or ""
        tool_calls: list[dict[str, Any]] = []

        # Check for function calls
        if hasattr(message, "function_call") and message.function_call:
            import json

            tool_calls.append(
                {
                    "name": message.function_call.name,
                    "input": json.loads(message.function_call.arguments),
                }
            )

        return LLMResponse(text=text, tool_calls=tool_calls, raw_response=response)


class LLMClientFactory:
    """Factory for creating LLM clients."""

    @staticmethod
    def create(
        provider: LLMProvider,
        api_key: str,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> BaseLLMClient:
        """
        Create an LLM client for the specified provider.

        Args:
            provider: Provider to use
            api_key: API key for the provider
            model: Optional model name (uses provider default if not specified)
            base_url: Optional custom base URL (for OpenAI-compatible endpoints)

        Returns:
            LLM client instance

        Raises:
            ValueError: If provider is not supported
        """
        if provider == LLMProvider.ANTHROPIC:
            if model:
                return AnthropicClient(api_key=api_key, model=model)
            return AnthropicClient(api_key=api_key)

        elif provider == LLMProvider.OPENAI:
            return OpenAIClient(
                api_key=api_key,
                model=model or "gpt-4",
                base_url=base_url
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")
