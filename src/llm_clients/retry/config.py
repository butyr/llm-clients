"""
Retry configuration and strategy definitions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Set


class RetryStrategy(str, Enum):
    """Available retry strategies."""

    EXPONENTIAL = "exponential"  # delay = base * (2 ** attempt)
    LINEAR = "linear"  # delay = base * attempt
    CONSTANT = "constant"  # delay = base


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 5)
        base_delay: Base delay in seconds (default: 1.0)
        max_delay: Maximum delay cap in seconds (default: 60.0)
        strategy: Backoff strategy to use (default: exponential)
        jitter: Jitter factor as fraction of delay (default: 0.25 = ±25%)
        retryable_status_codes: HTTP status codes that trigger retry
    """

    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: float = 0.25
    retryable_status_codes: Set[int] = field(
        default_factory=lambda: {429, 500, 502, 503, 504}
    )

    def should_retry(self, status_code: int) -> bool:
        """Check if the given status code should trigger a retry."""
        return status_code in self.retryable_status_codes

    @classmethod
    def aggressive(cls) -> "RetryConfig":
        """Preset for aggressive retry (more attempts, longer delays)."""
        return cls(
            max_retries=10,
            base_delay=2.0,
            max_delay=120.0,
        )

    @classmethod
    def conservative(cls) -> "RetryConfig":
        """Preset for conservative retry (fewer attempts, shorter delays)."""
        return cls(
            max_retries=3,
            base_delay=0.5,
            max_delay=10.0,
        )

    @classmethod
    def no_retry(cls) -> "RetryConfig":
        """Preset for no retry (single attempt only)."""
        return cls(max_retries=0)
