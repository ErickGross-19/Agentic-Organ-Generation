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
    LOCAL = "local"  # For local models via OpenAI-compatible API
    XAI = "xai"  # Grok models via xAI API (OpenAI-compatible)
    GROK = "grok"  # Alias for xAI
    GOOGLE = "google"  # Google Gemini models
    GEMINI = "gemini"  # Alias for Google
    MISTRAL = "mistral"  # Mistral AI models
    GROQ = "groq"  # Groq inference (OpenAI-compatible)


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
            provider = self.provider.lower()
            if provider == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY")
            elif provider == "anthropic":
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            elif provider in ("xai", "grok"):
                self.api_key = os.environ.get("XAI_API_KEY")
            elif provider in ("google", "gemini"):
                self.api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            elif provider == "mistral":
                self.api_key = os.environ.get("MISTRAL_API_KEY")
            elif provider == "groq":
                self.api_key = os.environ.get("GROQ_API_KEY")


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
        elif provider in ("xai", "grok"):
            return self._call_xai(messages, temperature, max_tokens)
        elif provider in ("google", "gemini"):
            return self._call_google(messages, temperature, max_tokens)
        elif provider == "mistral":
            return self._call_mistral(messages, temperature, max_tokens)
        elif provider == "groq":
            return self._call_groq(messages, temperature, max_tokens)
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
    
    def _call_xai(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call xAI/Grok API (OpenAI-compatible)."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        if self.config.api_key is None:
            raise ValueError("xAI API key not provided. Set XAI_API_KEY or pass api_key.")
        
        base_url = self.config.api_base or "https://api.x.ai/v1"
        
        client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=base_url,
        )
        
        model = self.config.model
        if model in ("gpt-4", "default"):
            model = "grok-beta"
        
        response = client.chat.completions.create(
            model=model,
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
    
    def _call_google(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call Google Gemini API."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
        
        if self.config.api_key is None:
            raise ValueError("Google API key not provided. Set GOOGLE_API_KEY or pass api_key.")
        
        genai.configure(api_key=self.config.api_key)
        
        model_name = self.config.model
        if model_name in ("gpt-4", "default"):
            model_name = "gemini-1.5-pro"
        
        model = genai.GenerativeModel(model_name)
        
        system_content = None
        chat_messages = []
        
        for m in messages:
            if m.role == "system":
                system_content = m.content
            elif m.role == "user":
                chat_messages.append({"role": "user", "parts": [m.content]})
            elif m.role == "assistant":
                chat_messages.append({"role": "model", "parts": [m.content]})
        
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        if system_content:
            model = genai.GenerativeModel(
                model_name,
                system_instruction=system_content,
            )
        
        chat = model.start_chat(history=chat_messages[:-1] if chat_messages else [])
        
        last_message = chat_messages[-1]["parts"][0] if chat_messages else ""
        response = chat.send_message(last_message, generation_config=generation_config)
        
        usage_metadata = getattr(response, 'usage_metadata', None)
        prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0) if usage_metadata else 0
        completion_tokens = getattr(usage_metadata, 'candidates_token_count', 0) if usage_metadata else 0
        
        return LLMResponse(
            content=response.text,
            model=model_name,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            finish_reason="stop",
            raw_response=None,
        )
    
    def _call_mistral(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call Mistral AI API."""
        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError("mistralai package not installed. Run: pip install mistralai")
        
        if self.config.api_key is None:
            raise ValueError("Mistral API key not provided. Set MISTRAL_API_KEY or pass api_key.")
        
        client = Mistral(api_key=self.config.api_key)
        
        model = self.config.model
        if model in ("gpt-4", "default"):
            model = "mistral-large-latest"
        
        response = client.chat.complete(
            model=model,
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
            raw_response=None,
        )
    
    def _call_groq(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call Groq API (OpenAI-compatible)."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        if self.config.api_key is None:
            raise ValueError("Groq API key not provided. Set GROQ_API_KEY or pass api_key.")
        
        base_url = self.config.api_base or "https://api.groq.com/openai/v1"
        
        client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=base_url,
        )
        
        model = self.config.model
        if model in ("gpt-4", "default"):
            model = "llama-3.3-70b-versatile"
        
        response = client.chat.completions.create(
            model=model,
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
