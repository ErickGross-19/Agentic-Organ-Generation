"""
LLM Healthcheck Module

Provides preflight checks and error handling for LLM configuration.
Prevents infinite loops when LLM isn't properly configured by performing
hard readiness checks before any workflow starts.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import time


# =============================================================================
# Error Types
# =============================================================================

class LLMError(Exception):
    """Base class for LLM-related errors."""
    pass


class MissingCredentialsError(LLMError):
    """
    Raised when required API credentials are missing.
    
    This error should stop execution immediately with setup instructions.
    """
    
    def __init__(self, provider: str, env_var: str, message: Optional[str] = None):
        self.provider = provider
        self.env_var = env_var
        if message is None:
            message = (
                f"Missing API credentials for provider '{provider}'. "
                f"Please set the {env_var} environment variable or pass api_key explicitly.\n"
                f"Example: export {env_var}=your-api-key"
            )
        super().__init__(message)


class ProviderMisconfiguredError(LLMError):
    """
    Raised when the LLM provider is misconfigured.
    
    This error should stop execution immediately.
    """
    
    def __init__(self, provider: str, issue: str, suggestion: Optional[str] = None):
        self.provider = provider
        self.issue = issue
        self.suggestion = suggestion
        message = f"Provider '{provider}' is misconfigured: {issue}"
        if suggestion:
            message += f"\nSuggestion: {suggestion}"
        super().__init__(message)


class TransientLLMError(LLMError):
    """
    Raised for transient errors like timeouts or rate limits.
    
    These errors should be retried with exponential backoff up to max retries.
    """
    
    def __init__(self, message: str, retry_after: Optional[float] = None):
        self.retry_after = retry_after
        super().__init__(message)


class FatalLLMError(LLMError):
    """
    Raised for fatal errors that cannot be recovered from.
    
    This error should stop execution immediately.
    """
    pass


class QuotaExhaustedError(FatalLLMError):
    """
    Raised when API quota is exhausted or billing is disabled.
    
    This is a FATAL error, not transient. A quota=0 or billing-disabled 429
    means the project cannot call the model at all, unlike a transient rate
    limit which can be retried.
    """
    
    def __init__(self, provider: str, message: Optional[str] = None):
        self.provider = provider
        if message is None:
            message = (
                f"API quota exhausted or billing disabled for provider '{provider}'. "
                f"This is a permanent error - the project cannot make API calls.\n"
                f"Please check your billing settings and quota limits."
            )
        super().__init__(message)


# =============================================================================
# Retry Configuration
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number (0-indexed)."""
        delay = self.initial_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)


DEFAULT_RETRY_CONFIG = RetryConfig()


# =============================================================================
# Provider Configuration
# =============================================================================

PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "xai": "XAI_API_KEY",
    "grok": "XAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "devin": "DEVIN_API_KEY",
    "local": None,  # Local doesn't require API key but needs api_base
}

# Alternative env vars that are also accepted for certain providers
PROVIDER_ALT_ENV_VARS = {
    "google": ["GEMINI_API_KEY"],
    "gemini": ["GEMINI_API_KEY"],
}

PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4",
    "anthropic": "claude-3-opus-20240229",
    "xai": "grok-beta",
    "grok": "grok-beta",
    "google": "gemini-1.5-pro",
    "gemini": "gemini-1.5-pro",
    "mistral": "mistral-large-latest",
    "groq": "llama-3.3-70b-versatile",
    "devin": "default",
    "local": "default",
}


# =============================================================================
# Healthcheck Functions
# =============================================================================

@dataclass
class HealthCheckResult:
    """Result of an LLM health check."""
    ready: bool
    provider: str
    model: str
    message: str
    ping_latency_ms: Optional[float] = None
    error: Optional[LLMError] = None


def check_credentials(provider: str, api_key: Optional[str] = None) -> None:
    """
    Check if credentials are available for the given provider.
    
    Parameters
    ----------
    provider : str
        LLM provider name
    api_key : str, optional
        Explicit API key (if not using environment variable)
        
    Raises
    ------
    MissingCredentialsError
        If credentials are not available
    ProviderMisconfiguredError
        If provider is not supported
    """
    provider_lower = provider.lower()
    
    if provider_lower not in PROVIDER_ENV_VARS:
        raise ProviderMisconfiguredError(
            provider=provider,
            issue=f"Unknown provider '{provider}'",
            suggestion=f"Supported providers: {', '.join(PROVIDER_ENV_VARS.keys())}"
        )
    
    env_var = PROVIDER_ENV_VARS[provider_lower]
    
    # Local provider doesn't need API key
    if provider_lower == "local":
        return
    
    # Check if API key is provided or available in environment
    if api_key:
        return
    
    # Check primary env var
    if env_var and os.environ.get(env_var):
        return
    
    # Check alternative env vars (e.g., GEMINI_API_KEY for google/gemini)
    alt_env_vars = PROVIDER_ALT_ENV_VARS.get(provider_lower, [])
    for alt_var in alt_env_vars:
        if os.environ.get(alt_var):
            return
    
    # Build helpful error message listing all accepted env vars
    all_vars = [env_var] if env_var else []
    all_vars.extend(alt_env_vars)
    env_var_str = " or ".join(all_vars) if all_vars else "API_KEY"
    
    raise MissingCredentialsError(provider=provider, env_var=env_var_str)


def check_provider_config(
    provider: str,
    api_base: Optional[str] = None,
    model: Optional[str] = None
) -> None:
    """
    Check if provider configuration is valid.
    
    Parameters
    ----------
    provider : str
        LLM provider name
    api_base : str, optional
        Custom API base URL
    model : str, optional
        Model name
        
    Raises
    ------
    ProviderMisconfiguredError
        If configuration is invalid
    """
    provider_lower = provider.lower()
    
    # Local provider requires api_base
    if provider_lower == "local" and not api_base:
        raise ProviderMisconfiguredError(
            provider=provider,
            issue="Local provider requires api_base to be set",
            suggestion="Set api_base to your local model server URL (e.g., http://localhost:8000/v1)"
        )


def ping_llm(
    provider: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 10.0
) -> float:
    """
    Perform a ping test to verify LLM connectivity.
    
    Parameters
    ----------
    provider : str
        LLM provider name
    api_key : str, optional
        API key
    api_base : str, optional
        Custom API base URL
    model : str, optional
        Model name
    timeout : float
        Timeout in seconds
        
    Returns
    -------
    float
        Latency in milliseconds
        
    Raises
    ------
    TransientLLMError
        If ping times out or fails transiently
    FatalLLMError
        If ping fails fatally
    """
    from .llm_client import LLMClient, LLMConfig
    
    provider_lower = provider.lower()
    
    # Use default model if not specified
    if not model:
        model = PROVIDER_DEFAULT_MODELS.get(provider_lower, "default")
    
    # Get API key from environment if not provided
    if not api_key:
        env_var = PROVIDER_ENV_VARS.get(provider_lower)
        if env_var:
            api_key = os.environ.get(env_var)
        # Check alternative env vars (e.g., GEMINI_API_KEY for google/gemini)
        if not api_key:
            alt_env_vars = PROVIDER_ALT_ENV_VARS.get(provider_lower, [])
            for alt_var in alt_env_vars:
                api_key = os.environ.get(alt_var)
                if api_key:
                    break
    
    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        api_base=api_base,
        max_tokens=10,
        temperature=0.0,
    )
    
    client = LLMClient(config=config)
    
    start_time = time.time()
    
    try:
        response = client.chat(
            message="Reply with exactly: OK",
            max_tokens=10,
            temperature=0.0,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Verify we got a response
        if not response.content:
            raise FatalLLMError("LLM returned empty response during ping")
        
        return latency_ms
        
    except Exception as e:
        error_str = str(e).lower()
        
        # Check for quota exhaustion / billing disabled errors FIRST
        # These are FATAL, not transient - the project cannot make API calls at all
        quota_indicators = [
            "quota",
            "billing",
            "limit=0",
            "requests limit 0",
            "free_tier_limit",
            "exceeded your current quota",
            "insufficient_quota",
            "billing_not_active",
            "billing hard limit",
            "you exceeded your current quota",
            "resource has been exhausted",
        ]
        if any(term in error_str for term in quota_indicators):
            raise QuotaExhaustedError(
                provider=provider,
                message=f"API quota exhausted or billing disabled: {e}"
            )
        
        # Check for transient rate limit errors (NOT quota exhaustion)
        # A transient 429 is "try again later"; quota=0 is "cannot call at all"
        transient_indicators = ["timeout", "503", "502", "temporarily unavailable", "overloaded"]
        rate_limit_transient = "429" in error_str and not any(q in error_str for q in quota_indicators)
        if any(term in error_str for term in transient_indicators) or rate_limit_transient:
            raise TransientLLMError(f"Ping failed with transient error: {e}")
        
        # Check for credential errors
        if any(term in error_str for term in ["unauthorized", "401", "invalid api key", "authentication"]):
            raise MissingCredentialsError(
                provider=provider,
                env_var=PROVIDER_ENV_VARS.get(provider_lower, "API_KEY"),
                message=f"Authentication failed: {e}"
            )
        
        # Check for configuration errors
        if any(term in error_str for term in ["not found", "404", "invalid model"]):
            raise ProviderMisconfiguredError(
                provider=provider,
                issue=str(e),
                suggestion="Check that the model name is correct and available"
            )
        
        # Default to fatal error
        raise FatalLLMError(f"Ping failed: {e}")


def assert_llm_ready(
    provider: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    skip_ping: bool = False,
    verbose: bool = True
) -> HealthCheckResult:
    """
    Assert that the LLM is ready for use.
    
    This function performs a complete health check including:
    1. Credential verification
    2. Configuration validation
    3. Connectivity ping (optional)
    
    Call this at the start of any workflow to prevent infinite loops
    when the LLM isn't properly configured.
    
    Parameters
    ----------
    provider : str
        LLM provider name
    api_key : str, optional
        API key
    api_base : str, optional
        Custom API base URL
    model : str, optional
        Model name
    skip_ping : bool
        If True, skip the ping test (faster but less thorough)
    verbose : bool
        If True, print status messages
        
    Returns
    -------
    HealthCheckResult
        Result of the health check
        
    Raises
    ------
    MissingCredentialsError
        If credentials are missing
    ProviderMisconfiguredError
        If provider is misconfigured
    FatalLLMError
        If ping fails fatally
    """
    provider_lower = provider.lower()
    model_name = model or PROVIDER_DEFAULT_MODELS.get(provider_lower, "default")
    
    if verbose:
        print(f"Checking LLM readiness for provider '{provider}'...")
    
    # Step 1: Check credentials
    try:
        check_credentials(provider, api_key)
        if verbose:
            print(f"  [OK] Credentials available")
    except LLMError as e:
        if verbose:
            print(f"  [FAIL] Credentials: {e}")
        raise
    
    # Step 2: Check configuration
    try:
        check_provider_config(provider, api_base, model)
        if verbose:
            print(f"  [OK] Configuration valid")
    except LLMError as e:
        if verbose:
            print(f"  [FAIL] Configuration: {e}")
        raise
    
    # Step 3: Ping test (optional)
    ping_latency = None
    if not skip_ping:
        try:
            ping_latency = ping_llm(provider, api_key, api_base, model)
            if verbose:
                print(f"  [OK] Ping successful ({ping_latency:.0f}ms)")
        except TransientLLMError as e:
            if verbose:
                print(f"  [WARN] Ping failed (transient): {e}")
            # Don't fail on transient errors during preflight
        except LLMError as e:
            if verbose:
                print(f"  [FAIL] Ping: {e}")
            raise
    
    if verbose:
        print(f"LLM ready: {provider} / {model_name}")
    
    return HealthCheckResult(
        ready=True,
        provider=provider,
        model=model_name,
        message="LLM is ready",
        ping_latency_ms=ping_latency,
    )


# =============================================================================
# Retry Decorator
# =============================================================================

def with_retry(
    func=None,
    *,
    config: Optional[RetryConfig] = None,
    on_transient_error: Optional[callable] = None
):
    """
    Decorator to add retry logic with exponential backoff.
    
    Only retries on TransientLLMError. Other errors are raised immediately.
    
    Parameters
    ----------
    func : callable
        Function to wrap
    config : RetryConfig, optional
        Retry configuration
    on_transient_error : callable, optional
        Callback called on each transient error with (attempt, error, delay)
        
    Returns
    -------
    callable
        Wrapped function with retry logic
        
    Examples
    --------
    >>> @with_retry(config=RetryConfig(max_retries=5))
    ... def call_llm(prompt):
    ...     return client.chat(prompt)
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG
    
    def decorator(fn):
        def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except TransientLLMError as e:
                    last_error = e
                    
                    if attempt < config.max_retries:
                        delay = e.retry_after or config.get_delay(attempt)
                        
                        if on_transient_error:
                            on_transient_error(attempt, e, delay)
                        
                        time.sleep(delay)
                    else:
                        # Max retries exceeded
                        raise FatalLLMError(
                            f"Max retries ({config.max_retries}) exceeded. "
                            f"Last error: {last_error}"
                        )
                except (MissingCredentialsError, ProviderMisconfiguredError, FatalLLMError):
                    # Don't retry on non-transient errors
                    raise
            
            # Should not reach here, but just in case
            raise FatalLLMError(f"Unexpected retry loop exit. Last error: {last_error}")
        
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator


# =============================================================================
# Convenience Functions
# =============================================================================

def get_provider_setup_instructions(provider: str) -> str:
    """
    Get setup instructions for a provider.
    
    Parameters
    ----------
    provider : str
        LLM provider name
        
    Returns
    -------
    str
        Setup instructions
    """
    provider_lower = provider.lower()
    env_var = PROVIDER_ENV_VARS.get(provider_lower)
    
    instructions = [f"Setup instructions for {provider}:"]
    
    if provider_lower == "openai":
        instructions.extend([
            "1. Get an API key from https://platform.openai.com/api-keys",
            f"2. Set the environment variable: export {env_var}=sk-...",
            "3. Or pass api_key='sk-...' when creating the client",
        ])
    elif provider_lower == "anthropic":
        instructions.extend([
            "1. Get an API key from https://console.anthropic.com/",
            f"2. Set the environment variable: export {env_var}=sk-ant-...",
            "3. Or pass api_key='sk-ant-...' when creating the client",
        ])
    elif provider_lower in ("xai", "grok"):
        instructions.extend([
            "1. Get an API key from https://x.ai/",
            f"2. Set the environment variable: export {env_var}=...",
            "3. Or pass api_key='...' when creating the client",
        ])
    elif provider_lower in ("google", "gemini"):
        instructions.extend([
            "1. Get an API key from https://makersuite.google.com/app/apikey",
            f"2. Set the environment variable: export {env_var}=... OR export GEMINI_API_KEY=...",
            "3. Or pass api_key='...' when creating the client",
        ])
    elif provider_lower == "mistral":
        instructions.extend([
            "1. Get an API key from https://console.mistral.ai/",
            f"2. Set the environment variable: export {env_var}=...",
            "3. Or pass api_key='...' when creating the client",
        ])
    elif provider_lower == "groq":
        instructions.extend([
            "1. Get an API key from https://console.groq.com/",
            f"2. Set the environment variable: export {env_var}=...",
            "3. Or pass api_key='...' when creating the client",
        ])
    elif provider_lower == "local":
        instructions.extend([
            "1. Start your local model server (e.g., llama.cpp, vLLM, etc.)",
            "2. Pass api_base='http://localhost:8000/v1' when creating the client",
            "3. Optionally set the model name with model='your-model-name'",
        ])
    else:
        instructions.append(f"Unknown provider. Check documentation for setup instructions.")
    
    return "\n".join(instructions)
