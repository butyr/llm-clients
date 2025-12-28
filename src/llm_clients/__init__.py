"""
Heimdall LLM Core - Reusable LLM/Agent Best Practices.

A collection of battle-tested patterns for building LLM-powered applications.
"""

from .clients import BaseLLMClient, Message, Role, OllamaClient, OpenRouterClient
from .exceptions import (
    LLMClientError,
    RateLimitError,
    ConnectionError,
    TimeoutError,
    AuthenticationError,
    ModelNotFoundError,
    InvalidRequestError,
    ServerError,
)
from .retry import RetryConfig, RetryStrategy, calculate_backoff

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Clients
    "BaseLLMClient",
    "Message",
    "Role",
    "OllamaClient",
    "OpenRouterClient",
    # Exceptions
    "LLMClientError",
    "RateLimitError",
    "ConnectionError",
    "TimeoutError",
    "AuthenticationError",
    "ModelNotFoundError",
    "InvalidRequestError",
    "ServerError",
    # Retry
    "RetryConfig",
    "RetryStrategy",
    "calculate_backoff",
]
