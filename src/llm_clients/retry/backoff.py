"""
Backoff calculation and retry decorators.
"""

import asyncio
import functools
import logging
import random
import time
from typing import Callable, TypeVar, ParamSpec, Awaitable

from .config import RetryConfig, RetryStrategy
from ..exceptions import LLMClientError

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def calculate_backoff(attempt: int, config: RetryConfig) -> float:
    """
    Calculate backoff delay for a given attempt.

    Args:
        attempt: Zero-based attempt number
        config: Retry configuration

    Returns:
        Delay in seconds with jitter applied
    """
    if config.strategy == RetryStrategy.EXPONENTIAL:
        delay = config.base_delay * (2**attempt)
    elif config.strategy == RetryStrategy.LINEAR:
        delay = config.base_delay * (attempt + 1)
    else:  # CONSTANT
        delay = config.base_delay

    # Apply max delay cap
    delay = min(delay, config.max_delay)

    # Apply jitter (±jitter%)
    if config.jitter > 0:
        jitter_amount = delay * config.jitter * (2 * random.random() - 1)
        delay = delay + jitter_amount

    return max(0, delay)


def with_retry(
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for synchronous functions with retry logic.

    Args:
        config: Retry configuration (default: RetryConfig())
        on_retry: Optional callback(attempt, exception, delay) called before each retry

    Returns:
        Decorated function with retry behavior
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except LLMClientError as e:
                    if not e.retryable or attempt >= config.max_retries:
                        raise
                    last_exception = e
                    delay = calculate_backoff(attempt, config)
                    if on_retry:
                        on_retry(attempt, e, delay)
                    else:
                        logger.warning(
                            f"Retry {attempt + 1}/{config.max_retries}: {e}, "
                            f"waiting {delay:.1f}s"
                        )
                    time.sleep(delay)
                except Exception as e:
                    # Non-LLMClientError exceptions are not retried
                    raise

            if last_exception:
                raise last_exception
            raise RuntimeError("Retry loop exited unexpectedly")

        return wrapper

    return decorator


def async_with_retry(
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Decorator for async functions with retry logic.

    Args:
        config: Retry configuration (default: RetryConfig())
        on_retry: Optional callback(attempt, exception, delay) called before each retry

    Returns:
        Decorated async function with retry behavior
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except LLMClientError as e:
                    if not e.retryable or attempt >= config.max_retries:
                        raise
                    last_exception = e
                    delay = calculate_backoff(attempt, config)
                    if on_retry:
                        on_retry(attempt, e, delay)
                    else:
                        logger.warning(
                            f"Retry {attempt + 1}/{config.max_retries}: {e}, "
                            f"waiting {delay:.1f}s"
                        )
                    await asyncio.sleep(delay)
                except Exception as e:
                    # Non-LLMClientError exceptions are not retried
                    raise

            if last_exception:
                raise last_exception
            raise RuntimeError("Retry loop exited unexpectedly")

        return wrapper

    return decorator
