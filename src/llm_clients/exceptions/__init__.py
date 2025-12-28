"""
Heimdall LLM Core - Exception Hierarchy.

Custom exceptions for LLM client operations with retry-awareness.
"""

from .base import (
    LLMClientError,
    RateLimitError,
    ConnectionError,
    TimeoutError,
    AuthenticationError,
    ModelNotFoundError,
    InvalidRequestError,
    ServerError,
)

__all__ = [
    "LLMClientError",
    "RateLimitError",
    "ConnectionError",
    "TimeoutError",
    "AuthenticationError",
    "ModelNotFoundError",
    "InvalidRequestError",
    "ServerError",
]
