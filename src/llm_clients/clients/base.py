"""
Base LLM client interface.

Defines the common interface that all LLM client adapters must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator

from ..retry import RetryConfig


class Role(str, Enum):
    """Message roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single message in a conversation."""

    role: Role
    content: str

    def to_dict(self) -> dict:
        """Convert to dictionary format for API requests."""
        return {"role": self.role.value, "content": self.content}

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Create an assistant message."""
        return cls(role=Role.ASSISTANT, content=content)


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.

    All LLM provider adapters must implement this interface.
    """

    def __init__(
        self,
        default_model: str | None = None,
        retry_config: RetryConfig | None = None,
        timeout: float = 120.0,
    ):
        """
        Initialize the client.

        Args:
            default_model: Default model to use if not specified per-request
            retry_config: Retry configuration for failed requests
            timeout: Request timeout in seconds
        """
        self.default_model = default_model
        self.retry_config = retry_config or RetryConfig()
        self.timeout = timeout

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name for logging."""
        ...

    def _build_messages(
        self,
        prompt: str,
        system: str | None = None,
        context: list[dict] | None = None,
    ) -> list[dict]:
        """
        Build message list from prompt, system message, and context.

        Args:
            prompt: The user prompt
            system: Optional system message
            context: Optional conversation history

        Returns:
            List of message dictionaries
        """
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        if context:
            messages.extend(context)

        messages.append({"role": "user", "content": prompt})

        return messages

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        context: list[dict] | None = None,
    ) -> str:
        """
        Generate a response (non-streaming).

        Args:
            prompt: The user prompt
            system: Optional system message
            model: Model to use (defaults to client's default_model)
            context: Optional conversation history

        Returns:
            The generated response text
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        context: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response.

        Args:
            prompt: The user prompt
            system: Optional system message
            model: Model to use (defaults to client's default_model)
            context: Optional conversation history

        Yields:
            Response text chunks as they arrive
        """
        ...

    @abstractmethod
    async def list_models(self) -> list[dict]:
        """
        List available models.

        Returns:
            List of model information dictionaries
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the service is accessible.

        Returns:
            True if healthy, False otherwise
        """
        ...
