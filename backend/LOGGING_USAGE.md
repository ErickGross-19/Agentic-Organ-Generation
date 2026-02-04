# Logging Usage Guide

## Quick Start

### Import the Logger
```python
from app.core.logging import get_logger

logger = get_logger(__name__)
```

### Basic Usage
```python
# Info level - normal operations
logger.info("Processing request")
logger.info(f"Generated scaffold in {time_ms:.2f}ms")

# Debug level - detailed diagnostics
logger.debug(f"Received parameters: {params}")
logger.debug(f"Internal state: count={count}, status={status}")

# Warning level - recoverable issues
logger.warning("API key not configured, using fallback mode")
logger.warning(f"Cache miss for scaffold {scaffold_id}")

# Error level - errors without stack trace
logger.error(f"Invalid parameters: {error_message}")
logger.error(f"Failed to initialize client: {e}")

# Exception level - errors WITH full stack trace
try:
    result = some_operation()
except Exception as e:
    logger.exception(f"Operation failed: {e}")
    raise
```

## Log Levels

| Level | When to Use | Example |
|-------|-------------|---------|
| `DEBUG` | Detailed diagnostic information | `logger.debug(f"Tool calls: {tool_calls}")` |
| `INFO` | General informational messages | `logger.info("Server started successfully")` |
| `WARNING` | Warning messages (recoverable) | `logger.warning("Using default configuration")` |
| `ERROR` | Error messages (no stack trace) | `logger.error("Validation failed")` |
| `EXCEPTION` | Errors with stack traces | `logger.exception("Unexpected error")` |

## Common Patterns

### API Endpoints
```python
from app.core.logging import get_logger

logger = get_logger(__name__)

@router.post("/generate")
async def generate_scaffold(request: GenerateRequest):
    logger.info(f"Generating scaffold: type={request.type}")

    try:
        result = _generate_scaffold(request.type, request.params)
        logger.info(f"Scaffold generated successfully")
        return result
    except ValueError as e:
        logger.error(f"Invalid parameters: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Service Initialization
```python
from app.core.logging import get_logger

logger = get_logger(__name__)

def initialize_service(config):
    logger.info(f"Initializing service with config: {config.name}")

    try:
        # ... initialization code ...
        logger.info("Service initialized successfully")
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        raise
```

### Long-Running Operations
```python
import time
from app.core.logging import get_logger

logger = get_logger(__name__)

def process_batch(items):
    logger.info(f"Processing batch of {len(items)} items")
    start_time = time.time()

    processed = 0
    for item in items:
        try:
            process_item(item)
            processed += 1
        except Exception as e:
            logger.error(f"Failed to process item {item.id}: {e}")

    duration_ms = (time.time() - start_time) * 1000
    logger.info(f"Batch processing complete: {processed}/{len(items)} items in {duration_ms:.2f}ms")
```

### Database Operations
```python
from app.core.logging import get_logger

logger = get_logger(__name__)

def save_to_database(data):
    logger.debug(f"Saving data: {data.id}")

    try:
        db.add(data)
        db.commit()
        logger.info(f"Data saved successfully: {data.id}")
    except Exception as e:
        logger.exception(f"Database save failed: {e}")
        db.rollback()
        raise
```

## Advanced Configuration

### Custom Log Levels
```python
from app.core.logging import get_logger
import logging

# Create logger with custom level
logger = get_logger(__name__, level=logging.DEBUG)
```

### File Logging
```python
from app.core.logging import setup_logger
from pathlib import Path

logger = setup_logger(
    __name__,
    level=logging.INFO,
    log_file=Path("/var/log/shed/app.log")
)
```

## Best Practices

### DO ✓
- Use `logger.info()` for normal operations
- Use `logger.exception()` in except blocks
- Include context in messages (IDs, types, counts)
- Use f-strings for formatted messages
- Log at appropriate levels

### DON'T ✗
- Use `print()` statements in production code
- Log sensitive data (passwords, API keys)
- Log at DEBUG level in production
- Create loggers in loops
- Catch exceptions without logging

## Examples by Module

### API Module
```python
from app.core.logging import get_logger

logger = get_logger(__name__)

# Request logging
logger.info(f"Request received: {request.method} {request.url}")

# Response logging
logger.info(f"Response sent: status={response.status_code}")

# Error logging
logger.error(f"Request failed: {error_message}")
```

### LLM Module
```python
from app.core.logging import get_logger

logger = get_logger(__name__)

# Initialization
logger.info(f"Initializing {provider} client with model {model}")

# API calls
logger.debug(f"Sending request to {provider} API")
logger.debug(f"Received {len(tool_calls)} tool calls")

# Errors
logger.error(f"{provider} API error: {error}")
```

### Database Module
```python
from app.core.logging import get_logger

logger = get_logger(__name__)

# Connection
logger.info(f"Database URL configured: {DATABASE_URL}")

# Operations
logger.debug(f"Executing query: {query}")
logger.info("Database tables initialized successfully")

# Errors
logger.exception(f"Database operation failed: {e}")
```

## Output Format

All logs follow this format:
```
YYYY-MM-DD HH:MM:SS - module.name - LEVEL - Message
```

Example:
```
2026-01-28 20:19:47 - app.api.scaffolds - INFO - Scaffold generated successfully in 245.67ms
2026-01-28 20:19:48 - app.llm.agent - ERROR - API error: Connection timeout
```

## Environment Variables

Control logging behavior with environment variables:

```bash
# Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
export LOG_LEVEL=INFO

# Enable file logging
export LOG_FILE=/var/log/shed/app.log

# Log format (simple, detailed, json)
export LOG_FORMAT=detailed
```

Note: These require updates to the logging configuration.

## Troubleshooting

### Logs not appearing
Check that the logger is initialized:
```python
from app.core.logging import get_logger
logger = get_logger(__name__)  # Must be at module level
```

### Too many debug messages
Adjust log level:
```python
logger = get_logger(__name__, level=logging.INFO)
```

### Missing stack traces
Use `logger.exception()` instead of `logger.error()`:
```python
try:
    risky_operation()
except Exception as e:
    logger.exception(f"Operation failed: {e}")  # Includes stack trace
```
