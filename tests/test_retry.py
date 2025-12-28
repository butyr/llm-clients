"""Tests for retry module - behavior focused."""

import pytest
from llm_clients.retry import RetryConfig, RetryStrategy, calculate_backoff


class TestCalculateBackoff:
    """Test backoff calculation behavior."""

    def test_backoff_increases_with_attempts(self):
        """Given increasing attempts, delay should grow."""
        config = RetryConfig(base_delay=1.0, jitter=0)

        delay_0 = calculate_backoff(0, config)
        delay_1 = calculate_backoff(1, config)
        delay_2 = calculate_backoff(2, config)

        assert delay_0 < delay_1 < delay_2

    def test_backoff_respects_max_delay(self):
        """Delay never exceeds max_delay config."""
        config = RetryConfig(base_delay=1.0, max_delay=5.0, jitter=0)

        # Even at high attempt numbers, should not exceed max
        delay = calculate_backoff(100, config)

        assert delay <= config.max_delay

    def test_backoff_with_zero_jitter_is_deterministic(self):
        """When jitter=0, delay is deterministic."""
        config = RetryConfig(base_delay=1.0, jitter=0)

        delay_a = calculate_backoff(2, config)
        delay_b = calculate_backoff(2, config)

        assert delay_a == delay_b

    def test_backoff_with_jitter_varies(self):
        """When jitter > 0, delays should vary (probabilistic)."""
        config = RetryConfig(base_delay=1.0, jitter=0.25)

        # Run multiple times - at least some should differ
        delays = [calculate_backoff(2, config) for _ in range(20)]
        unique_delays = set(delays)

        # With 25% jitter, we expect variation
        assert len(unique_delays) > 1

    def test_linear_strategy_grows_linearly(self):
        """Linear strategy: delay = base * (attempt + 1)."""
        config = RetryConfig(
            base_delay=1.0, strategy=RetryStrategy.LINEAR, jitter=0
        )

        delay_0 = calculate_backoff(0, config)
        delay_1 = calculate_backoff(1, config)
        delay_2 = calculate_backoff(2, config)

        # Linear growth: 1, 2, 3
        assert delay_0 == 1.0
        assert delay_1 == 2.0
        assert delay_2 == 3.0

    def test_constant_strategy_stays_constant(self):
        """Constant strategy: delay = base always."""
        config = RetryConfig(
            base_delay=2.0, strategy=RetryStrategy.CONSTANT, jitter=0
        )

        delay_0 = calculate_backoff(0, config)
        delay_5 = calculate_backoff(5, config)
        delay_10 = calculate_backoff(10, config)

        assert delay_0 == delay_5 == delay_10 == 2.0


class TestRetryConfig:
    """Test RetryConfig behavior."""

    def test_should_retry_rate_limit_status(self):
        """429 status should trigger retry."""
        config = RetryConfig()
        assert config.should_retry(429) is True

    def test_should_retry_server_errors(self):
        """5xx statuses should trigger retry."""
        config = RetryConfig()

        assert config.should_retry(500) is True
        assert config.should_retry(502) is True
        assert config.should_retry(503) is True
        assert config.should_retry(504) is True

    def test_should_not_retry_client_errors(self):
        """4xx statuses (except 429) should not trigger retry."""
        config = RetryConfig()

        assert config.should_retry(400) is False
        assert config.should_retry(401) is False
        assert config.should_retry(403) is False
        assert config.should_retry(404) is False

    def test_aggressive_preset_has_more_retries(self):
        """Aggressive preset should have more retries than default."""
        default = RetryConfig()
        aggressive = RetryConfig.aggressive()

        assert aggressive.max_retries > default.max_retries

    def test_conservative_preset_has_fewer_retries(self):
        """Conservative preset should have fewer retries than default."""
        default = RetryConfig()
        conservative = RetryConfig.conservative()

        assert conservative.max_retries < default.max_retries

    def test_no_retry_preset_has_zero_retries(self):
        """No retry preset should have zero retries."""
        no_retry = RetryConfig.no_retry()

        assert no_retry.max_retries == 0
