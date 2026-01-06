"""
LLM Client

Client for interacting with LLM APIs (OpenAI, Anthropic, etc.).
Provides a unified interface for sending prompts and receiving responses.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import json
import os


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # For local models


@dataclass
class LLMConfig:
    """
    Configuration for LLM client.
    
    Attributes
    ----------
    provider : str
        LLM provider: "openai", "anthropic", or "local"
    model : str
        Model name (e.g., "gpt-4", "claude-3-opus")
    api_key : str, optional
        API key (can also be set via environment variable)
    api_base : str, optional
        Custom API base URL (for local models or proxies)
    max_tokens : int
        Maximum tokens in response
    temperature : float
        Sampling temperature (0.0 = deterministic, 1.0 = creative)
    system_prompt : str, optional
        System prompt to prepend to all conversations
    """
    provider: str = "openai"
    model: str = "gpt-4"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    
    def __post_init__(self):
        # Try to get API key from environment if not provided
        if self.api_key is None:
            if self.provider == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY")
            elif self.provider == "anthropic":
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")


@dataclass
class Message:
    """A message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from LLM API."""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    raw_response: Optional[Dict[str, Any]] = None


class LLMClient:
    """
    Client for interacting with LLM APIs.
    
    Provides a unified interface for sending prompts and receiving responses
    from various LLM providers.
    
    Parameters
    ----------
    config : LLMConfig, optional
        Client configuration. If not provided, uses defaults.
    provider : str, optional
        LLM provider (shortcut for config.provider)
    api_key : str, optional
        API key (shortcut for config.api_key)
    model : str, optional
        Model name (shortcut for config.model)
        
    Examples
    --------
    >>> from automation.llm_client import LLMClient
    >>> 
    >>> # Using OpenAI
    >>> client = LLMClient(provider="openai", api_key="sk-...")
    >>> response = client.chat("Generate a vascular network design spec")
    >>> print(response.content)
    >>> 
    >>> # Using Anthropic
    >>> client = LLMClient(provider="anthropic", model="claude-3-opus")
    >>> response = client.chat("Analyze this network structure")
    """
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        if config is None:
            config = LLMConfig()
        
        # Override config with explicit parameters
        if provider is not None:
            config.provider = provider
        if api_key is not None:
            config.api_key = api_key
        if model is not None:
            config.model = model
        
        self.config = config
        self._conversation_history: List[Message] = []
        
        # Initialize system prompt if provided
        if config.system_prompt:
            self._conversation_history.append(
                Message(role="system", content=config.system_prompt)
            )
    
    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        continue_conversation: bool = False,
    ) -> LLMResponse:
        """
        Send a message and get a response.
        
        Parameters
        ----------
        message : str
            User message to send
        system_prompt : str, optional
            Override system prompt for this request
        temperature : float, optional
            Override temperature for this request
        max_tokens : int, optional
            Override max_tokens for this request
        continue_conversation : bool
            If True, include previous conversation history
            
        Returns
        -------
        LLMResponse
            Response from the LLM
        """
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        elif self.config.system_prompt and not continue_conversation:
            messages.append(Message(role="system", content=self.config.system_prompt))
        
        # Add conversation history if continuing
        if continue_conversation:
            messages.extend(self._conversation_history)
        
        # Add user message
        messages.append(Message(role="user", content=message))
        
        # Make API call
        response = self._call_api(
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )
        
        # Update conversation history
        if continue_conversation:
            self._conversation_history.append(Message(role="user", content=message))
            self._conversation_history.append(Message(role="assistant", content=response.content))
        
        return response
    
    def _call_api(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Make the actual API call based on provider."""
        provider = self.config.provider.lower()
        
        if provider == "openai":
            return self._call_openai(messages, temperature, max_tokens)
        elif provider == "anthropic":
            return self._call_anthropic(messages, temperature, max_tokens)
        elif provider == "local":
            return self._call_local(messages, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _call_openai(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call OpenAI API."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        if self.config.api_key is None:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key.")
        
        client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base,
        )
        
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=response.choices[0].finish_reason,
            raw_response=response.model_dump(),
        )
    
    def _call_anthropic(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call Anthropic API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        if self.config.api_key is None:
            raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY or pass api_key.")
        
        client = anthropic.Anthropic(api_key=self.config.api_key)
        
        # Anthropic uses a different message format
        system_content = None
        api_messages = []
        
        for m in messages:
            if m.role == "system":
                system_content = m.content
            else:
                api_messages.append({"role": m.role, "content": m.content})
        
        kwargs = {
            "model": self.config.model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if system_content:
            kwargs["system"] = system_content
        
        response = client.messages.create(**kwargs)
        
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            finish_reason=response.stop_reason,
            raw_response=response.model_dump(),
        )
    
    def _call_local(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call local model API (OpenAI-compatible)."""
        # Use OpenAI client with custom base URL
        try:
            import openai
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        if self.config.api_base is None:
            raise ValueError("api_base must be set for local provider")
        
        client = openai.OpenAI(
            api_key=self.config.api_key or "not-needed",
            base_url=self.config.api_base,
        )
        
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                "total_tokens": getattr(response.usage, 'total_tokens', 0),
            },
            finish_reason=response.choices[0].finish_reason,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
        )
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history = []
        if self.config.system_prompt:
            self._conversation_history.append(
                Message(role="system", content=self.config.system_prompt)
            )
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history as list of dicts."""
        return [{"role": m.role, "content": m.content} for m in self._conversation_history]
