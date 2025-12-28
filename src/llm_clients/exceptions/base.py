"""
Base exception classes for LLM client operations.

Each exception includes a `retryable` flag indicating whether the operation
can be safely retried with the same parameters.
"""


class LLMClientError(Exception):
    """Base exception for all LLM client errors."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        provider: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.status_code = status_code
        self.provider = provider

    def __str__(self) -> str:
        parts = [self.message]
        if self.provider:
            parts.insert(0, f"[{self.provider}]")
        if self.status_code:
            parts.append(f"(status: {self.status_code})")
        return " ".join(parts)


class RateLimitError(LLMClientError):
    """Raised when rate limit is exceeded. Always retryable."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        retry_after: float | None = None,
        **kwargs,
    ):
        super().__init__(message, retryable=True, **kwargs)
        self.retry_after = retry_after


class ConnectionError(LLMClientError):
    """Raised when connection to the LLM service fails. Usually retryable."""

    def __init__(self, message: str = "Connection failed", **kwargs):
        super().__init__(message, retryable=True, **kwargs)


class TimeoutError(LLMClientError):
    """Raised when request times out. Usually retryable."""

    def __init__(self, message: str = "Request timed out", **kwargs):
        super().__init__(message, retryable=True, **kwargs)


class AuthenticationError(LLMClientError):
    """Raised when authentication fails. Not retryable."""

    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, retryable=False, **kwargs)


class ModelNotFoundError(LLMClientError):
    """Raised when the requested model is not available. Not retryable."""

    def __init__(self, message: str = "Model not found", model: str | None = None, **kwargs):
        super().__init__(message, retryable=False, **kwargs)
        self.model = model


class InvalidRequestError(LLMClientError):
    """Raised when the request is malformed. Not retryable."""

    def __init__(self, message: str = "Invalid request", **kwargs):
        super().__init__(message, retryable=False, **kwargs)


class ServerError(LLMClientError):
    """Raised when the server returns a 5xx error. Usually retryable."""

    def __init__(self, message: str = "Server error", **kwargs):
        super().__init__(message, retryable=True, **kwargs)
