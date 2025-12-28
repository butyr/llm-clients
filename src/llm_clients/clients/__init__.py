"""
Heimdall LLM Core - LLM Clients.

Unified interface for multiple LLM providers.
"""

from .base import BaseLLMClient, Message, Role
from .ollama import OllamaClient
from .openrouter import OpenRouterClient

__all__ = [
    "BaseLLMClient",
    "Message",
    "Role",
    "OllamaClient",
    "OpenRouterClient",
]
