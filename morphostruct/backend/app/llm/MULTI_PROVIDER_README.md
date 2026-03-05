# Multi-Provider LLM Abstraction

The SHED backend now supports multiple LLM providers through a unified abstraction layer.

## Supported Providers

- **Anthropic** (Claude) - Default provider
- **OpenAI** (GPT-4, GPT-3.5)

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# Choose your provider ("anthropic" or "openai")
LLM_PROVIDER=anthropic

# API Keys (provide the one for your chosen provider)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional: Override default model
LLM_MODEL=claude-sonnet-4-20250514  # For Anthropic
# LLM_MODEL=gpt-4                   # For OpenAI
```

### Default Models

If `LLM_MODEL` is not specified, the following defaults are used:

- **Anthropic**: `claude-sonnet-4-20250514`
- **OpenAI**: `gpt-4`

## Architecture

### Core Components

1. **`providers.py`**: Multi-provider abstraction layer
   - `BaseLLMClient`: Abstract interface for all providers
   - `AnthropicClient`: Anthropic/Claude implementation
   - `OpenAIClient`: OpenAI implementation
   - `LLMClientFactory`: Factory for creating provider instances
   - `LLMMessage`: Provider-agnostic message format
   - `LLMResponse`: Provider-agnostic response format

2. **`agent.py`**: Updated to use multi-provider abstraction
   - Automatically selects provider based on config
   - Converts messages to provider-agnostic format
   - Handles responses uniformly across providers

3. **`config.py`**: Updated settings
   - `llm_provider`: Provider selection
   - `anthropic_api_key`: Anthropic API key
   - `openai_api_key`: OpenAI API key
   - `llm_model`: Optional model override

## Usage

### Using the ScaffoldAgent

The `ScaffoldAgent` automatically uses the configured provider:

```python
from app.llm.agent import get_agent

agent = get_agent()
response = await agent.chat(
    message="Create a bone scaffold",
    conversation_history=[],
)
```

### Programmatic Provider Selection

You can also specify the provider programmatically:

```python
from app.llm.agent import ScaffoldAgent

# Use Anthropic
agent = ScaffoldAgent(
    provider="anthropic",
    api_key="sk-ant-...",
)

# Use OpenAI
agent = ScaffoldAgent(
    provider="openai",
    api_key="sk-...",
    model="gpt-4",
)
```

### Direct Client Usage

For advanced use cases, you can use the clients directly:

```python
from app.llm.providers import LLMClientFactory, LLMProvider, LLMMessage

# Create an Anthropic client
client = LLMClientFactory.create(
    provider=LLMProvider.ANTHROPIC,
    api_key="sk-ant-...",
)

# Send a message
messages = [LLMMessage(role="user", content="Hello!")]
response = await client.chat_async(
    messages=messages,
    system="You are a helpful assistant.",
)

print(response.text)
```

## Adding New Providers

To add a new provider:

1. **Add provider enum** in `providers.py`:
   ```python
   class LLMProvider(str, Enum):
       ANTHROPIC = "anthropic"
       OPENAI = "openai"
       YOUR_PROVIDER = "your_provider"  # Add this
   ```

2. **Create client class** that extends `BaseLLMClient`:
   ```python
   class YourProviderClient(BaseLLMClient):
       def __init__(self, api_key: str, model: str = "default-model"):
           # Initialize your provider's client
           pass

       def chat(self, messages, system, tools, max_tokens, temperature):
           # Implement sync chat
           pass

       async def chat_async(self, messages, system, tools, max_tokens, temperature):
           # Implement async chat
           pass
   ```

3. **Update factory** in `LLMClientFactory.create()`:
   ```python
   elif provider == LLMProvider.YOUR_PROVIDER:
       if model:
           return YourProviderClient(api_key=api_key, model=model)
       return YourProviderClient(api_key=api_key)
   ```

4. **Update config** in `config.py`:
   ```python
   your_provider_api_key: str = ""
   ```

5. **Update agent** in `agent.py` to handle the new API key:
   ```python
   elif self.provider == LLMProvider.YOUR_PROVIDER:
       self.api_key = settings.your_provider_api_key
   ```

## Tool Calling

Both providers support tool/function calling, but with different formats:

- **Anthropic**: Uses `tools` parameter with `tool_use` responses
- **OpenAI**: Uses `functions` parameter with `function_call` responses

The abstraction layer automatically converts between formats, so your code works with both providers seamlessly.

## Error Handling

All providers handle errors gracefully and return fallback responses when the LLM is unavailable.

## Dependencies

- **Anthropic**: `anthropic>=0.18.0` (already installed)
- **OpenAI**: `openai` (install with `pip install openai`)

## Testing

To test provider switching:

1. Set up both API keys in `.env`
2. Switch `LLM_PROVIDER` between `anthropic` and `openai`
3. Restart the backend server
4. Send a chat message and verify the response

## Migration from Legacy Code

Old code that directly used `anthropic.Anthropic` can be gradually migrated:

**Before:**
```python
import anthropic
client = anthropic.Anthropic(api_key=api_key)
response = client.messages.create(...)
```

**After:**
```python
from app.llm.providers import LLMClientFactory, LLMProvider, LLMMessage
client = LLMClientFactory.create(LLMProvider.ANTHROPIC, api_key)
response = await client.chat_async(messages=[...])
```

The `ScaffoldAgent` already uses the new abstraction, so no changes are needed there.
