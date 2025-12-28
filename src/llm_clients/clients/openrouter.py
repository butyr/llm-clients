"""
OpenRouter LLM client adapter.

OpenRouter provides access to multiple LLM providers through a unified API.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator

import httpx

from .base import BaseLLMClient
from ..exceptions import (
    AuthenticationError,
    ConnectionError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ModelNotFoundError,
    InvalidRequestError,
)
from ..retry import RetryConfig, calculate_backoff

logger = logging.getLogger(__name__)


class OpenRouterClient(BaseLLMClient):
    """
    Client for OpenRouter unified LLM API.

    Features:
    - Access to multiple cloud LLM providers
    - Rate limiting with automatic retry
    - Streaming and non-streaming generation
    - Exponential backoff with jitter
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        default_model: str = "openai/gpt-4o-mini",
        retry_config: RetryConfig | None = None,
        timeout: float = 120.0,
        app_name: str = "Heimdall LLM",
        app_url: str = "http://localhost:3000",
    ):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key
            base_url: OpenRouter API base URL
            default_model: Default model to use
            retry_config: Retry configuration
            timeout: Request timeout in seconds
            app_name: Application name for X-Title header
            app_url: Application URL for HTTP-Referer header
        """
        super().__init__(default_model, retry_config, timeout)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.app_name = app_name
        self.app_url = app_url

    @property
    def provider_name(self) -> str:
        return "OpenRouter"

    def _get_headers(self) -> dict:
        """Get headers for OpenRouter API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.app_url,
            "X-Title": self.app_name,
        }

    def _handle_error(self, status_code: int, response_text: str = "") -> None:
        """Convert HTTP status codes to domain exceptions."""
        if status_code == 401:
            raise AuthenticationError(
                "Invalid API key",
                provider=self.provider_name,
                status_code=status_code,
            )
        elif status_code == 404:
            raise ModelNotFoundError(
                "Model not found",
                provider=self.provider_name,
                status_code=status_code,
            )
        elif status_code == 400:
            raise InvalidRequestError(
                f"Invalid request: {response_text}",
                provider=self.provider_name,
                status_code=status_code,
            )
        elif status_code == 429:
            raise RateLimitError(
                "Rate limit exceeded",
                provider=self.provider_name,
                status_code=status_code,
            )
        elif status_code >= 500:
            raise ServerError(
                f"Server error: {response_text}",
                provider=self.provider_name,
                status_code=status_code,
            )

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        context: list[dict] | None = None,
    ) -> str:
        """Generate a response from OpenRouter (non-streaming) with retry."""
        model = model or self.default_model
        messages = self._build_messages(prompt, system, context)

        last_exception: Exception | None = None

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._get_headers(),
                        json={"model": model, "messages": messages, "stream": False},
                    )

                    # Check for retryable status codes
                    if self.retry_config.should_retry(response.status_code):
                        delay = calculate_backoff(attempt, self.retry_config)
                        logger.warning(
                            f"[{self.provider_name}] Rate limited (status {response.status_code}), "
                            f"retrying in {delay:.1f}s ({attempt + 1}/{self.retry_config.max_retries})"
                        )
                        await asyncio.sleep(delay)
                        continue

                    if response.status_code != 200:
                        self._handle_error(response.status_code, response.text)

                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"]

            except httpx.HTTPStatusError as e:
                if self.retry_config.should_retry(e.response.status_code):
                    delay = calculate_backoff(attempt, self.retry_config)
                    logger.warning(
                        f"[{self.provider_name}] Request failed (status {e.response.status_code}), "
                        f"retrying in {delay:.1f}s ({attempt + 1}/{self.retry_config.max_retries})"
                    )
                    last_exception = e
                    await asyncio.sleep(delay)
                    continue
                self._handle_error(e.response.status_code, e.response.text)

            except httpx.ConnectError as e:
                last_exception = ConnectionError(
                    f"Failed to connect to OpenRouter",
                    provider=self.provider_name,
                )
                if attempt < self.retry_config.max_retries:
                    delay = calculate_backoff(attempt, self.retry_config)
                    logger.warning(
                        f"[{self.provider_name}] Connection error, "
                        f"retrying in {delay:.1f}s ({attempt + 1}/{self.retry_config.max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise last_exception

            except httpx.ReadTimeout as e:
                last_exception = TimeoutError(
                    f"Request timed out after {self.timeout}s",
                    provider=self.provider_name,
                )
                if attempt < self.retry_config.max_retries:
                    delay = calculate_backoff(attempt, self.retry_config)
                    logger.warning(
                        f"[{self.provider_name}] Timeout, "
                        f"retrying in {delay:.1f}s ({attempt + 1}/{self.retry_config.max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise last_exception

        # All retries exhausted
        logger.error(f"[{self.provider_name}] All {self.retry_config.max_retries} retries exhausted")
        if last_exception:
            raise last_exception
        raise RuntimeError("OpenRouter request failed after all retries")

    async def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        context: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from OpenRouter with retry."""
        model = model or self.default_model
        messages = self._build_messages(prompt, system, context)

        logger.info(f"[{self.provider_name}] Starting stream with model: {model}")

        last_exception: Exception | None = None

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/chat/completions",
                        headers=self._get_headers(),
                        json={"model": model, "messages": messages, "stream": True},
                    ) as response:
                        # Check for rate limiting before streaming
                        if self.retry_config.should_retry(response.status_code):
                            delay = calculate_backoff(attempt, self.retry_config)
                            logger.warning(
                                f"[{self.provider_name}] Rate limited (status {response.status_code}), "
                                f"retrying in {delay:.1f}s ({attempt + 1}/{self.retry_config.max_retries})"
                            )
                            await asyncio.sleep(delay)
                            continue

                        if response.status_code != 200:
                            self._handle_error(response.status_code)

                        response.raise_for_status()

                        # Stream tokens
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:]
                                if data_str.strip() == "[DONE]":
                                    return  # Success
                                try:
                                    data = json.loads(data_str)
                                    if "choices" in data and len(data["choices"]) > 0:
                                        delta = data["choices"][0].get("delta", {})
                                        content = delta.get("content")
                                        if content:
                                            yield content
                                except json.JSONDecodeError:
                                    continue
                        return  # Success

            except httpx.HTTPStatusError as e:
                if self.retry_config.should_retry(e.response.status_code):
                    delay = calculate_backoff(attempt, self.retry_config)
                    logger.warning(
                        f"[{self.provider_name}] Stream failed (status {e.response.status_code}), "
                        f"retrying in {delay:.1f}s ({attempt + 1}/{self.retry_config.max_retries})"
                    )
                    last_exception = e
                    await asyncio.sleep(delay)
                    continue
                self._handle_error(e.response.status_code, e.response.text)

            except httpx.ConnectError as e:
                last_exception = ConnectionError(
                    f"Failed to connect to OpenRouter",
                    provider=self.provider_name,
                )
                if attempt < self.retry_config.max_retries:
                    delay = calculate_backoff(attempt, self.retry_config)
                    logger.warning(
                        f"[{self.provider_name}] Stream connection error, "
                        f"retrying in {delay:.1f}s ({attempt + 1}/{self.retry_config.max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise last_exception

            except httpx.ReadTimeout as e:
                last_exception = TimeoutError(
                    f"Stream timed out after {self.timeout}s",
                    provider=self.provider_name,
                )
                if attempt < self.retry_config.max_retries:
                    delay = calculate_backoff(attempt, self.retry_config)
                    logger.warning(
                        f"[{self.provider_name}] Stream timeout, "
                        f"retrying in {delay:.1f}s ({attempt + 1}/{self.retry_config.max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise last_exception

        # All retries exhausted
        logger.error(f"[{self.provider_name}] All {self.retry_config.max_retries} stream retries exhausted")
        if last_exception:
            raise last_exception
        raise RuntimeError("OpenRouter stream failed after all retries")

    async def list_models(self) -> list[dict]:
        """List available models from OpenRouter."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/models",
                headers=self._get_headers(),
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])

    async def health_check(self) -> bool:
        """Check if OpenRouter API is accessible."""
        if not self.api_key:
            logger.warning(f"[{self.provider_name}] API key not configured")
            return False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self._get_headers(),
                )
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"[{self.provider_name}] Health check failed: {e}")
            return False
