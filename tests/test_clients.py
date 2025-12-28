"""Tests for LLM clients - behavior focused with HTTP mocking."""

import json
import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from llm_clients.clients import OllamaClient, OpenRouterClient
from llm_clients.exceptions import (
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
)
from llm_clients.retry import RetryConfig


# --- Helper to create proper mock responses ---


def create_response(status_code: int, json_data: dict | None = None, text: str = "") -> httpx.Response:
    """Create a mock response with a proper request object."""
    request = httpx.Request("POST", "http://test")
    if json_data:
        return httpx.Response(status_code, json=json_data, request=request)
    return httpx.Response(status_code, text=text, request=request)


# --- Fixtures ---


@pytest.fixture
def ollama_client():
    """Create Ollama client with no retries for predictable tests."""
    return OllamaClient(
        base_url="http://localhost:11434",
        default_model="llama3.2:3b",
        retry_config=RetryConfig.no_retry(),
    )


@pytest.fixture
def openrouter_client():
    """Create OpenRouter client with no retries for predictable tests."""
    return OpenRouterClient(
        api_key="test-api-key",
        default_model="openai/gpt-4o-mini",
        retry_config=RetryConfig.no_retry(),
    )


# --- Helper to create mock responses ---


def mock_ollama_response(content: str) -> dict:
    """Create a mock Ollama chat response."""
    return {"message": {"role": "assistant", "content": content}}


def mock_openrouter_response(content: str) -> dict:
    """Create a mock OpenRouter chat response."""
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def mock_stream_lines(chunks: list[str], provider: str = "ollama") -> list[str]:
    """Create mock streaming response lines."""
    lines = []
    for chunk in chunks:
        if provider == "ollama":
            lines.append(json.dumps({"message": {"content": chunk}}))
        else:  # openrouter
            lines.append(f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}")
    if provider == "openrouter":
        lines.append("data: [DONE]")
    return lines


# --- Ollama Client Tests ---


class TestOllamaGenerate:
    """Test Ollama generate behavior."""

    @pytest.mark.asyncio
    async def test_returns_content_on_success(self, ollama_client):
        """Given 200 response, returns message content."""
        mock_response = create_response(200, mock_ollama_response("Hello, world!"))

        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            result = await ollama_client.generate("Hi")

            assert result == "Hello, world!"

    @pytest.mark.asyncio
    async def test_raises_model_not_found_on_404(self, ollama_client):
        """Given 404, raises ModelNotFoundError."""
        mock_response = create_response(404, text="model not found")

        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            with pytest.raises(ModelNotFoundError):
                await ollama_client.generate("Hi")

    @pytest.mark.asyncio
    async def test_uses_provided_model(self, ollama_client):
        """When model is specified, it should be used in request."""
        mock_response = create_response(200, mock_ollama_response("Response"))

        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            await ollama_client.generate("Hi", model="custom-model")

            # Check the model was passed in the request
            call_kwargs = mock_post.call_args
            request_json = call_kwargs.kwargs.get("json", {})
            assert request_json.get("model") == "custom-model"


class TestOllamaHealthCheck:
    """Test Ollama health check behavior."""

    @pytest.mark.asyncio
    async def test_returns_true_when_up(self, ollama_client):
        """Given 200, returns True."""
        mock_response = create_response(200, {"models": []})

        with patch.object(
            httpx.AsyncClient, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = mock_response

            result = await ollama_client.health_check()

            assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_down(self, ollama_client):
        """Given connection error, returns False."""
        with patch.object(
            httpx.AsyncClient, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")

            result = await ollama_client.health_check()

            assert result is False


class TestOllamaModelManagement:
    """Test Ollama model unloading behavior."""

    @pytest.mark.asyncio
    async def test_unloads_previous_model_on_switch(self):
        """When switching models, previous model should be unloaded."""
        client = OllamaClient(
            retry_config=RetryConfig.no_retry(),
            auto_unload=True,
        )
        client._current_model = "old-model"

        mock_response = create_response(200, mock_ollama_response("Response"))

        unload_calls = []

        async def track_post(*args, **kwargs):
            url = args[0] if args else kwargs.get("url", "")
            json_data = kwargs.get("json", {})

            # Track unload calls (keep_alive: 0)
            if json_data.get("keep_alive") == 0:
                unload_calls.append(json_data.get("model"))
                return create_response(200, {})

            return mock_response

        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.side_effect = track_post

            await client.generate("Hi", model="new-model")

            # Should have unloaded the old model
            assert "old-model" in unload_calls


# --- OpenRouter Client Tests ---


class TestOpenRouterGenerate:
    """Test OpenRouter generate behavior."""

    @pytest.mark.asyncio
    async def test_returns_content_on_success(self, openrouter_client):
        """Given 200 response, returns message content."""
        mock_response = create_response(200, mock_openrouter_response("Hello from OpenRouter!"))

        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            result = await openrouter_client.generate("Hi")

            assert result == "Hello from OpenRouter!"

    @pytest.mark.asyncio
    async def test_raises_auth_error_on_401(self, openrouter_client):
        """Given 401, raises AuthenticationError immediately."""
        mock_response = create_response(401, text="Invalid API key")

        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            with pytest.raises(AuthenticationError):
                await openrouter_client.generate("Hi")

    @pytest.mark.asyncio
    async def test_includes_auth_header(self, openrouter_client):
        """Request should include Authorization header."""
        mock_response = create_response(200, mock_openrouter_response("Response"))

        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            await openrouter_client.generate("Hi")

            call_kwargs = mock_post.call_args
            headers = call_kwargs.kwargs.get("headers", {})
            assert "Authorization" in headers
            assert "test-api-key" in headers["Authorization"]


class TestOpenRouterRetry:
    """Test OpenRouter retry behavior."""

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit(self):
        """Given 429 then 200, eventually succeeds."""
        client = OpenRouterClient(
            api_key="test-key",
            retry_config=RetryConfig(max_retries=2, base_delay=0.01),
        )

        responses = [
            create_response(429, text="Rate limited"),
            create_response(200, mock_openrouter_response("Success!")),
        ]
        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            response = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            return response

        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post_obj:
            mock_post_obj.side_effect = mock_post

            result = await client.generate("Hi")

            assert result == "Success!"
            assert call_count == 2  # First failed, second succeeded

    @pytest.mark.asyncio
    async def test_does_not_retry_auth_error(self):
        """Given 401, should not retry."""
        client = OpenRouterClient(
            api_key="bad-key",
            retry_config=RetryConfig(max_retries=3, base_delay=0.01),
        )

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return create_response(401, text="Unauthorized")

        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock
        ) as mock_post_obj:
            mock_post_obj.side_effect = mock_post

            with pytest.raises(AuthenticationError):
                await client.generate("Hi")

            # Should only try once (no retries for auth errors)
            assert call_count == 1


class TestOpenRouterHealthCheck:
    """Test OpenRouter health check behavior."""

    @pytest.mark.asyncio
    async def test_returns_true_when_up(self, openrouter_client):
        """Given 200, returns True."""
        mock_response = create_response(200, {"data": []})

        with patch.object(
            httpx.AsyncClient, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = mock_response

            result = await openrouter_client.health_check()

            assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_without_api_key(self):
        """Without API key, returns False."""
        client = OpenRouterClient(api_key="")

        result = await client.health_check()

        assert result is False


# --- Shared Interface Tests ---


class TestClientInterface:
    """Test that both clients implement the same interface."""

    @pytest.mark.asyncio
    async def test_both_have_generate(self, ollama_client, openrouter_client):
        """Both clients should have generate method."""
        assert hasattr(ollama_client, "generate")
        assert hasattr(openrouter_client, "generate")
        assert callable(ollama_client.generate)
        assert callable(openrouter_client.generate)

    @pytest.mark.asyncio
    async def test_both_have_generate_stream(self, ollama_client, openrouter_client):
        """Both clients should have generate_stream method."""
        assert hasattr(ollama_client, "generate_stream")
        assert hasattr(openrouter_client, "generate_stream")

    @pytest.mark.asyncio
    async def test_both_have_health_check(self, ollama_client, openrouter_client):
        """Both clients should have health_check method."""
        assert hasattr(ollama_client, "health_check")
        assert hasattr(openrouter_client, "health_check")

    @pytest.mark.asyncio
    async def test_both_have_list_models(self, ollama_client, openrouter_client):
        """Both clients should have list_models method."""
        assert hasattr(ollama_client, "list_models")
        assert hasattr(openrouter_client, "list_models")
