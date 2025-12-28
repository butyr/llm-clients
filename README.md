# LLM Clients

Reusable LLM/Agent best practices extracted from production projects.

## Installation

```bash
pip install llm-clients
```

Or for development:

```bash
pip install -e ".[dev]"
```

## Usage

### LLM Clients

Both clients implement the same interface (`BaseLLMClient`), making them interchangeable.

#### Ollama (Local Inference)

```python
from llm_clients import OllamaClient

client = OllamaClient(
    base_url="http://localhost:11434",
    default_model="llama3.2:3b",
)

# Non-streaming
response = await client.generate(
    prompt="What is Python?",
    system="You are a helpful assistant.",
)

# Streaming
async for chunk in client.generate_stream(prompt="Tell me a story"):
    print(chunk, end="", flush=True)
```

#### OpenRouter (Cloud Models)

```python
from llm_clients import OpenRouterClient

client = OpenRouterClient(
    api_key="sk-or-...",
    default_model="openai/gpt-4o-mini",
)

response = await client.generate(
    prompt="Explain quantum computing",
    system="You are a physics professor.",
)
```

### Retry Configuration

Customize retry behavior for both clients:

```python
from llm_clients import RetryConfig, RetryStrategy, OllamaClient

# Custom configuration
config = RetryConfig(
    max_retries=10,
    base_delay=2.0,
    max_delay=120.0,
    strategy=RetryStrategy.EXPONENTIAL,
    jitter=0.25,
)

client = OllamaClient(retry_config=config)

# Or use presets
client = OllamaClient(retry_config=RetryConfig.aggressive())
client = OllamaClient(retry_config=RetryConfig.conservative())
client = OllamaClient(retry_config=RetryConfig.no_retry())
```

### Exception Handling

All exceptions include a `retryable` flag:

```python
from llm_clients import (
    LLMClientError,
    RateLimitError,
    ConnectionError,
    TimeoutError,
    AuthenticationError,
)

try:
    response = await client.generate(prompt="Hello")
except RateLimitError as e:
    print(f"Rate limited: {e}")  # e.retryable == True
except AuthenticationError as e:
    print(f"Auth failed: {e}")  # e.retryable == False
except LLMClientError as e:
    if e.retryable:
        print("Can retry this error")
```

### Health Checks

```python
if await client.health_check():
    print("Service is available")
else:
    print("Service is down")
```

### List Available Models

```python
models = await client.list_models()
for model in models:
    print(model["name"])
```

## Architecture

```
llm_clients/
├── clients/          # LLM client adapters
│   ├── base.py       # Abstract base class
│   ├── ollama.py     # Ollama adapter
│   └── openrouter.py # OpenRouter adapter
├── exceptions/       # Custom exception hierarchy
│   └── base.py       # Exception classes
└── retry/            # Retry logic
    ├── config.py     # RetryConfig dataclass
    └── backoff.py    # Backoff calculations
```

## Dual Provider Architecture

This library supports a dual-provider setup:

- **Ollama**: Local inference with automatic GPU/CPU detection. Use for development and when you have local GPU resources.
- **OpenRouter**: Cloud access to larger models (GPT-4, Claude, etc.). Use when you need more powerful models that won't fit locally.

These are **complementary**, not fallbacks. Choose based on your needs:

```python
# Local development with smaller model
ollama = OllamaClient(default_model="llama3.2:3b")

# Cloud for powerful reasoning
openrouter = OpenRouterClient(
    api_key="...",
    default_model="anthropic/claude-3-opus",
)
```

## License

MIT
