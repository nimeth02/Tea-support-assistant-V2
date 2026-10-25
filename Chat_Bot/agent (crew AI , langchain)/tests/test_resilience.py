"""Circuit breaker states and the "retry only if the request never arrived" rule."""
import httpx
import pytest

from agent.ops.resilience import CircuitBreaker, is_connect_error, retry_on_connect_error


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_circuit_opens_after_threshold_and_recovers():
    clock = FakeClock()
    breaker = CircuitBreaker("t1", failure_threshold=3, reset_seconds=30, clock=clock)

    for _ in range(2):
        breaker.record_failure()
    assert breaker.state == "closed" and breaker.allow()

    breaker.record_failure()  # third failure in a row
    assert breaker.state == "open" and not breaker.allow()

    clock.now = 31  # wait is over: one trial call is allowed
    assert breaker.state == "half_open"
    assert breaker.allow()
    assert not breaker.allow()  # only one trial at a time

    breaker.record_success()
    assert breaker.state == "closed" and breaker.allow()


def test_failed_trial_reopens_circuit():
    clock = FakeClock()
    breaker = CircuitBreaker("t2", failure_threshold=1, reset_seconds=10, clock=clock)
    breaker.record_failure()
    clock.now = 11
    assert breaker.allow()
    breaker.record_failure()
    assert breaker.state == "open"


def test_success_resets_failure_count():
    breaker = CircuitBreaker("t3", failure_threshold=2)
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.state == "closed"


def test_connect_error_detected_through_wrapping():
    try:
        try:
            raise httpx.ConnectError("refused")
        except httpx.ConnectError as inner:
            raise RuntimeError("A2A client error") from inner
    except RuntimeError as wrapped:
        assert is_connect_error(wrapped)
    assert not is_connect_error(httpx.ReadTimeout("slow"))
    assert not is_connect_error(ValueError("bad"))


@pytest.mark.anyio
async def test_retries_connect_errors_then_succeeds():
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectError("refused")
        return "ok"

    assert await retry_on_connect_error(flaky, name="t4", retries=2, backoff=0) == "ok"
    assert len(calls) == 3


@pytest.mark.anyio
async def test_does_not_retry_once_request_was_delivered():
    calls = []

    async def slow():
        calls.append(1)
        raise httpx.ReadTimeout("expert is slow")  # the expert got the question: don't resend

    with pytest.raises(httpx.ReadTimeout):
        await retry_on_connect_error(slow, name="t5", retries=3, backoff=0)
    assert len(calls) == 1
