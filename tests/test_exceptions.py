"""Tests for exceptions module - behavior focused."""

import pytest
from llm_clients.exceptions import (
    LLMClientError,
    RateLimitError,
    ConnectionError,
    TimeoutError,
    AuthenticationError,
    ModelNotFoundError,
    InvalidRequestError,
    ServerError,
)


class TestRetryableFlag:
    """Test that exceptions have correct retryable defaults."""

    def test_rate_limit_error_is_retryable(self):
        """Rate limit errors should be retryable."""
        error = RateLimitError()
        assert error.retryable is True

    def test_connection_error_is_retryable(self):
        """Connection errors should be retryable."""
        error = ConnectionError()
        assert error.retryable is True

    def test_timeout_error_is_retryable(self):
        """Timeout errors should be retryable."""
        error = TimeoutError()
        assert error.retryable is True

    def test_server_error_is_retryable(self):
        """Server errors should be retryable."""
        error = ServerError()
        assert error.retryable is True

    def test_auth_error_not_retryable(self):
        """Authentication errors should not be retryable."""
        error = AuthenticationError()
        assert error.retryable is False

    def test_model_not_found_not_retryable(self):
        """Model not found errors should not be retryable."""
        error = ModelNotFoundError()
        assert error.retryable is False

    def test_invalid_request_not_retryable(self):
        """Invalid request errors should not be retryable."""
        error = InvalidRequestError()
        assert error.retryable is False

    def test_base_error_not_retryable_by_default(self):
        """Base LLMClientError should not be retryable by default."""
        error = LLMClientError("test")
        assert error.retryable is False


class TestExceptionStringRepresentation:
    """Test that exception string includes useful context."""

    def test_str_includes_message(self):
        """String representation should include the message."""
        error = LLMClientError("Something went wrong")
        assert "Something went wrong" in str(error)

    def test_str_includes_provider_when_set(self):
        """String representation should include provider name."""
        error = LLMClientError("Error", provider="OpenRouter")
        assert "OpenRouter" in str(error)

    def test_str_includes_status_code_when_set(self):
        """String representation should include status code."""
        error = LLMClientError("Error", status_code=429)
        assert "429" in str(error)

    def test_str_includes_all_context(self):
        """String should include provider, message, and status."""
        error = LLMClientError(
            "Rate limited",
            provider="Ollama",
            status_code=429,
        )
        result = str(error)

        assert "Ollama" in result
        assert "Rate limited" in result
        assert "429" in result


class TestRateLimitErrorExtras:
    """Test RateLimitError specific behavior."""

    def test_retry_after_is_stored(self):
        """retry_after value should be accessible."""
        error = RateLimitError(retry_after=30.0)
        assert error.retry_after == 30.0

    def test_retry_after_defaults_to_none(self):
        """retry_after should default to None."""
        error = RateLimitError()
        assert error.retry_after is None


class TestModelNotFoundErrorExtras:
    """Test ModelNotFoundError specific behavior."""

    def test_model_name_is_stored(self):
        """Model name should be accessible."""
        error = ModelNotFoundError(model="gpt-5-turbo")
        assert error.model == "gpt-5-turbo"

    def test_model_defaults_to_none(self):
        """Model should default to None."""
        error = ModelNotFoundError()
        assert error.model is None


class TestExceptionInheritance:
    """Test that all exceptions inherit from LLMClientError."""

    @pytest.mark.parametrize(
        "exception_class",
        [
            RateLimitError,
            ConnectionError,
            TimeoutError,
            AuthenticationError,
            ModelNotFoundError,
            InvalidRequestError,
            ServerError,
        ],
    )
    def test_inherits_from_base(self, exception_class):
        """All exception types should be catchable as LLMClientError."""
        error = exception_class()
        assert isinstance(error, LLMClientError)
