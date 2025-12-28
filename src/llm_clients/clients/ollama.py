"""
Ollama LLM client adapter.

Ollama provides local LLM inference with automatic GPU/CPU detection.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator

import httpx

from .base import BaseLLMClient
from ..exceptions import (
    ConnectionError,
    ServerError,
    TimeoutError,
    ModelNotFoundError,
)
from ..retry import RetryConfig, calculate_backoff

logger = logging.getLogger(__name__)


class OllamaClient(BaseLLMClient):
    """
    Client for Ollama local LLM inference.

    Features:
    - Automatic GPU/CPU detection by Ollama
    - Model memory management (unload unused models)
    - Streaming and non-streaming generation
    - Retry with exponential backoff
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama3.2:3b",
        retry_config: RetryConfig | None = None,
        timeout: float = 120.0,
        auto_unload: bool = True,
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama API base URL
            default_model: Default model to use
            retry_config: Retry configuration
            timeout: Request timeout in seconds
            auto_unload: Automatically unload previous model when switching
        """
        super().__init__(default_model, retry_config, timeout)
        self.base_url = base_url.rstrip("/")
        self.auto_unload = auto_unload
        self._current_model: str | None = None

    @property
    def provider_name(self) -> str:
        return "Ollama"

    async def unload_model(self, model: str) -> None:
        """Unload a model from memory by setting keep_alive to 0."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": model, "keep_alive": 0},
                )
                logger.info(f"[{self.provider_name}] Unloaded model: {model}")
        except Exception as e:
            logger.warning(f"[{self.provider_name}] Failed to unload model {model}: {e}")

    async def unload_all_models(self) -> None:
        """Unload all currently loaded models."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/ps")
                if response.status_code == 200:
                    data = response.json()
                    for model_info in data.get("models", []):
                        model_name = model_info.get("name")
                        if model_name:
                            await self.unload_model(model_name)
        except Exception as e:
            logger.warning(f"[{self.provider_name}] Failed to unload models: {e}")

    async def _ensure_model_loaded(self, model: str) -> None:
        """Unload previous model before loading a new one to save memory."""
        if self.auto_unload and self._current_model and self._current_model != model:
            logger.info(
                f"[{self.provider_name}] Switching from {self._current_model} to {model}"
            )
            await self.unload_model(self._current_model)
        self._current_model = model

    def _handle_error(self, e: httpx.HTTPStatusError) -> None:
        """Convert HTTP errors to domain exceptions."""
        status = e.response.status_code
        if status == 404:
            raise ModelNotFoundError(
                f"Model not found",
                provider=self.provider_name,
                status_code=status,
            )
        elif status >= 500:
            raise ServerError(
                f"Server error: {e.response.text}",
                provider=self.provider_name,
                status_code=status,
            )
        raise e

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        context: list[dict] | None = None,
    ) -> str:
        """Generate a response from Ollama (non-streaming) with retry."""
        model = model or self.default_model
        await self._ensure_model_loaded(model)
        messages = self._build_messages(prompt, system, context)

        last_exception: Exception | None = None

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/api/chat",
                        json={"model": model, "messages": messages, "stream": False},
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data["message"]["content"]

            except httpx.HTTPStatusError as e:
                self._handle_error(e)

            except httpx.ConnectError as e:
                last_exception = ConnectionError(
                    f"Failed to connect to Ollama at {self.base_url}",
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

        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected retry loop exit")

    async def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        context: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from Ollama with retry."""
        model = model or self.default_model
        await self._ensure_model_loaded(model)
        messages = self._build_messages(prompt, system, context)

        last_exception: Exception | None = None

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/api/chat",
                        json={"model": model, "messages": messages, "stream": True},
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if line:
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    yield data["message"]["content"]
                        return  # Success

            except httpx.HTTPStatusError as e:
                self._handle_error(e)

            except httpx.ConnectError as e:
                last_exception = ConnectionError(
                    f"Failed to connect to Ollama at {self.base_url}",
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

        if last_exception:
            raise last_exception

    async def list_models(self) -> list[dict]:
        """List available models from Ollama."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])

    async def health_check(self) -> bool:
        """Check if Ollama is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"[{self.provider_name}] Health check failed: {e}")
            return False
