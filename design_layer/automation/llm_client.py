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
import time


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
    DEVIN = "devin"  # Devin AI via session-based API


def _safe_int(value, default: int = 0) -> int:
    """
    Safely convert a value to int, returning default if None or invalid.
    
    This is used for defensive token counting where SDK responses may have
    None values or missing fields.
    
    Parameters
    ----------
    value : Any
        Value to convert to int
    default : int
        Default value if conversion fails (default: 0)
        
    Returns
    -------
    int
        The converted value or default
    """
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_gemini_text(response) -> tuple:
    """
    Extract text content from a Gemini API response.
    
    The google.genai SDK can return text in different places depending on
    the model, response type, and safety filtering. This function tries
    multiple extraction strategies.
    
    Parameters
    ----------
    response : GenerateContentResponse
        Response from google.genai generate_content call
        
    Returns
    -------
    tuple
        (text_content, finish_reason, diagnostic_info)
        - text_content: str - The extracted text (may be empty)
        - finish_reason: str - The finish reason (stop, safety, etc.)
        - diagnostic_info: dict - Debug info about extraction
    """
    text_content = ""
    finish_reason = "unknown"
    diagnostic_info = {
        "extraction_method": None,
        "candidates_count": 0,
        "has_safety_ratings": False,
        "safety_blocked": False,
        "block_reason": None,
    }
    
    # Strategy 1: Try response.text (convenience property)
    try:
        if hasattr(response, 'text') and response.text:
            text_content = response.text
            diagnostic_info["extraction_method"] = "response.text"
    except (ValueError, AttributeError):
        # response.text can raise ValueError if no valid candidate
        pass
    
    # Strategy 2: Extract from candidates[0].content.parts
    if not text_content:
        candidates = getattr(response, 'candidates', None) or []
        diagnostic_info["candidates_count"] = len(candidates) if candidates else 0
        
        if candidates:
            candidate = candidates[0]
            
            # Get finish reason from candidate
            candidate_finish = getattr(candidate, 'finish_reason', None)
            if candidate_finish:
                # Convert enum to string if needed
                finish_reason = str(candidate_finish).lower()
                if hasattr(candidate_finish, 'name'):
                    finish_reason = candidate_finish.name.lower()
            
            # Check for safety blocking
            safety_ratings = getattr(candidate, 'safety_ratings', None)
            if safety_ratings:
                diagnostic_info["has_safety_ratings"] = True
                for rating in safety_ratings:
                    blocked = getattr(rating, 'blocked', False)
                    if blocked:
                        diagnostic_info["safety_blocked"] = True
                        category = getattr(rating, 'category', 'unknown')
                        diagnostic_info["block_reason"] = str(category)
                        break
            
            # Extract text from content.parts
            content = getattr(candidate, 'content', None)
            if content:
                parts = getattr(content, 'parts', None) or []
                text_parts = []
                for part in parts:
                    part_text = getattr(part, 'text', None)
                    if part_text:
                        text_parts.append(part_text)
                if text_parts:
                    text_content = "".join(text_parts)
                    diagnostic_info["extraction_method"] = "candidates[0].content.parts"
    
    # Strategy 3: Check prompt_feedback for blocking
    if not text_content:
        prompt_feedback = getattr(response, 'prompt_feedback', None)
        if prompt_feedback:
            block_reason = getattr(prompt_feedback, 'block_reason', None)
            if block_reason:
                diagnostic_info["safety_blocked"] = True
                diagnostic_info["block_reason"] = str(block_reason)
                finish_reason = "blocked"
    
    # Default finish reason if we got text but no explicit reason
    if text_content and finish_reason == "unknown":
        finish_reason = "stop"
    
    return text_content, finish_reason, diagnostic_info


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
    max_retries : int
        Maximum number of retries for transient errors (default: 3)
    retry_delay : float
        Initial delay between retries in seconds (default: 1.0)
    retry_max_delay : float
        Maximum delay between retries in seconds (default: 30.0)
    """
    provider: str = "openai"
    model: str = "gpt-4"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_max_delay: float = 30.0
    
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
            elif provider == "devin":
                self.api_key = os.environ.get("DEVIN_API_KEY")


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
        api_base: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
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
        if api_base is not None:
            config.api_base = api_base
        if temperature is not None:
            config.temperature = temperature
        if max_tokens is not None:
            config.max_tokens = max_tokens
        
        self.config = config
        self._conversation_history: List[Message] = []
        self._devin_session_id: Optional[str] = None  # Track Devin session for conversation continuity
        
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
        """Make the actual API call based on provider with retry logic."""
        provider = self.config.provider.lower()
        
        # Get the appropriate provider method
        provider_methods = {
            "openai": self._call_openai,
            "anthropic": self._call_anthropic,
            "local": self._call_local,
            "xai": self._call_xai,
            "grok": self._call_xai,
            "google": self._call_google,
            "gemini": self._call_google,
            "mistral": self._call_mistral,
            "groq": self._call_groq,
            "devin": self._call_devin,
        }
        
        if provider not in provider_methods:
            raise ValueError(f"Unsupported provider: {provider}")
        
        method = provider_methods[provider]
        
        # Retry logic with exponential backoff
        last_error = None
        delay = self.config.retry_delay
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return method(messages, temperature, max_tokens)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if this is a transient error that should be retried
                is_transient = any(term in error_str for term in [
                    "timeout", "rate limit", "429", "503", "502", "504",
                    "connection", "temporary", "overloaded", "capacity"
                ])
                
                # Check if this is a fatal error that should not be retried
                is_fatal = any(term in error_str for term in [
                    "unauthorized", "401", "invalid api key", "authentication",
                    "not found", "404", "invalid model", "permission denied"
                ])
                
                if is_fatal or attempt >= self.config.max_retries:
                    # Don't retry fatal errors or if we've exhausted retries
                    raise
                
                if is_transient:
                    # Wait before retrying with exponential backoff
                    time.sleep(delay)
                    delay = min(delay * 2, self.config.retry_max_delay)
                else:
                    # Unknown error type, raise immediately
                    raise
        
        # Should not reach here, but just in case
        raise last_error if last_error else RuntimeError("Unexpected retry loop exit")
    
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
        """Call Google Gemini API using the new Google GenAI SDK.
        
        Supported Gemini models include:
        - gemini-3-flash-preview (latest preview)
        - gemini-3-flash (latest generation)
        - gemini-2.5-flash (recommended default)
        - gemini-2.5-flash-lite (faster, lower cost)
        - gemini-2.5-flash-tts (text-to-speech capable)
        - gemini-2.0-flash (previous generation)
        - gemini-1.5-pro (previous generation)
        
        Any valid Gemini model name can be passed via config.model.
        """
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("google-genai package not installed. Run: pip install google-genai")
        
        if self.config.api_key is None:
            raise ValueError("Google API key not provided. Set GOOGLE_API_KEY or GEMINI_API_KEY or pass api_key.")
        
        client = genai.Client(api_key=self.config.api_key)
        
        model_name = self.config.model
        if model_name in ("gpt-4", "default"):
            model_name = "gemini-2.5-flash"
        
        system_content = None
        contents = []
        
        for m in messages:
            if m.role == "system":
                system_content = m.content
            elif m.role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=m.content)]))
            elif m.role == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part(text=m.content)]))
        
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        if system_content:
            config.system_instruction = system_content
        
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )
        
        # Extract text content using robust extraction (handles empty response.text)
        text_content, finish_reason, diagnostic_info = _extract_gemini_text(response)
        
        # If content is empty and response was blocked, raise a clear error
        if not text_content and diagnostic_info.get("safety_blocked"):
            block_reason = diagnostic_info.get("block_reason", "unknown")
            raise ValueError(
                f"Gemini response blocked by safety filter: {block_reason}. "
                f"Candidates: {diagnostic_info.get('candidates_count', 0)}"
            )
        
        # Defensive token parsing: handle None values and different SDK field names
        # The google.genai SDK may return None for token counts or use different field names
        usage_metadata = getattr(response, 'usage_metadata', None)
        
        # Try multiple attribute names for prompt tokens (compatibility shim)
        prompt_tokens = 0
        if usage_metadata:
            prompt_tokens = _safe_int(
                getattr(usage_metadata, 'prompt_token_count', None) or
                getattr(usage_metadata, 'input_token_count', None),
                0
            )
        
        # Try multiple attribute names for completion tokens (compatibility shim)
        completion_tokens = 0
        if usage_metadata:
            completion_tokens = _safe_int(
                getattr(usage_metadata, 'candidates_token_count', None) or
                getattr(usage_metadata, 'output_token_count', None),
                0
            )
        
        # Try to get total from SDK, otherwise compute it
        total_tokens = prompt_tokens + completion_tokens
        if usage_metadata:
            sdk_total = _safe_int(getattr(usage_metadata, 'total_token_count', None), 0)
            if sdk_total > 0:
                total_tokens = sdk_total
        
        return LLMResponse(
            content=text_content,
            model=model_name,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            finish_reason=finish_reason,
            raw_response={"diagnostic_info": diagnostic_info},
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
    
    def _call_devin(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """
        Call Devin API via session-based interface.
        
        Devin uses a session-based model rather than stateless chat completions.
        This method creates a new session or sends a message to an existing session,
        then polls for completion and returns the result.
        
        Parameters
        ----------
        messages : List[Message]
            Messages to send (system prompt + user message)
        temperature : float
            Not used by Devin API (included for interface compatibility)
        max_tokens : int
            Not used by Devin API (included for interface compatibility)
            
        Returns
        -------
        LLMResponse
            Response from Devin
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests package not installed. Run: pip install requests")
        
        if self.config.api_key is None:
            raise ValueError("Devin API key not provided. Set DEVIN_API_KEY or pass api_key.")
        
        base_url = self.config.api_base or "https://api.devin.ai/v1"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        
        # Build the prompt from messages
        prompt_parts = []
        for m in messages:
            if m.role == "system":
                prompt_parts.append(f"[System Instructions]\n{m.content}\n")
            elif m.role == "user":
                prompt_parts.append(m.content)
            elif m.role == "assistant":
                prompt_parts.append(f"[Previous Response]\n{m.content}\n")
        
        prompt = "\n".join(prompt_parts)
        
        # Decide whether to create a new session or send message to existing one
        if self._devin_session_id is None:
            # Create a new session
            response = requests.post(
                f"{base_url}/sessions",
                headers=headers,
                json={
                    "prompt": prompt,
                    "idempotent": False,
                },
            )
            
            if response.status_code != 200:
                raise ValueError(f"Devin API error creating session: {response.status_code} - {response.text}")
            
            session_data = response.json()
            self._devin_session_id = session_data.get("session_id")
            
            if not self._devin_session_id:
                raise ValueError(f"Devin API did not return session_id: {session_data}")
        else:
            # Send message to existing session
            response = requests.post(
                f"{base_url}/sessions/{self._devin_session_id}/message",
                headers=headers,
                json={"message": prompt},
            )
            
            if response.status_code != 200:
                # Session may have expired, try creating a new one
                self._devin_session_id = None
                return self._call_devin(messages, temperature, max_tokens)
        
        # Poll for completion
        import time
        max_poll_time = 300  # 5 minutes max
        poll_interval = 2  # Start with 2 seconds
        max_poll_interval = 30  # Max 30 seconds between polls
        elapsed = 0
        
        while elapsed < max_poll_time:
            status_response = requests.get(
                f"{base_url}/sessions/{self._devin_session_id}",
                headers=headers,
            )
            
            if status_response.status_code != 200:
                raise ValueError(f"Devin API error polling session: {status_response.status_code} - {status_response.text}")
            
            status_data = status_response.json()
            status_enum = status_data.get("status_enum", "")
            
            if status_enum in ("blocked", "stopped"):
                # Session is complete, extract response
                structured_output = status_data.get("structured_output", {})
                
                # Try to get response content from various possible fields
                content = (
                    structured_output.get("response") or
                    structured_output.get("output") or
                    structured_output.get("result") or
                    status_data.get("last_message", {}).get("content") or
                    str(structured_output) if structured_output else "Task completed."
                )
                
                return LLMResponse(
                    content=content,
                    model="devin",
                    usage={
                        "prompt_tokens": 0,  # Devin doesn't report token usage
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    finish_reason="stop" if status_enum == "stopped" else "blocked",
                    raw_response=status_data,
                )
            
            # Wait before next poll with exponential backoff
            time.sleep(poll_interval)
            elapsed += poll_interval
            poll_interval = min(poll_interval * 1.5, max_poll_interval)
        
        # Timeout reached
        raise TimeoutError(f"Devin session {self._devin_session_id} did not complete within {max_poll_time} seconds")
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history = []
        self._devin_session_id = None  # Reset Devin session when clearing history
        if self.config.system_prompt:
            self._conversation_history.append(
                Message(role="system", content=self.config.system_prompt)
            )
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history as list of dicts."""
        return [{"role": m.role, "content": m.content} for m in self._conversation_history]
