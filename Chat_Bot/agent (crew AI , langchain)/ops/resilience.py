"""Retries and circuit breakers for calls between agents.

Why: in a multi-agent system each expert is a separate server. Any of them can be restarting,
overloaded, or out of LLM quota. Without protection, one slow expert makes every chat wait for
the full timeout, and the orchestrator keeps hammering a server that is already down.

  retry_on_connect_error  try again (with a growing wait) ONLY when the request never reached
                          the expert. If the expert did receive the question, we do not resend
                          it: that would run the LLM twice and double the cost.

  CircuitBreaker          like a fuse in a house. After N failures in a row it "opens" and calls
                          to that expert fail instantly for a while, instead of waiting for a
                          timeout. After the wait it lets one test call through ("half-open");
                          success closes it again, failure opens it again.
"""
import asyncio
import logging
import time
from typing import Awaitable, Callable, TypeVar

import httpx

from agent.ops.telemetry import CIRCUIT_STATE, CLIENT_RETRIES

logger = logging.getLogger("agent.resilience")
T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    """Raised instead of calling an expert whose circuit is open."""


class CircuitBreaker:
    CLOSED, HALF_OPEN, OPEN = "closed", "half_open", "open"
    _GAUGE = {CLOSED: 0, HALF_OPEN: 1, OPEN: 2}

    def __init__(self, name: str, failure_threshold: int = 3, reset_seconds: float = 30.0,
                 clock: Callable[[], float] = time.monotonic):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_seconds = reset_seconds
        self._clock = clock
        self._failures = 0
        self._opened_at = 0.0
        self._state = self.CLOSED
        self._trial_running = False
        CIRCUIT_STATE.labels(name).set(0)

    @property
    def state(self) -> str:
        # An open circuit becomes half-open by itself once the wait is over.
        if self._state == self.OPEN and self._clock() - self._opened_at >= self.reset_seconds:
            self._set(self.HALF_OPEN)
        return self._state

    def allow(self) -> bool:
        """May we call the expert now? In half-open state only one trial call at a time."""
        state = self.state
        if state == self.CLOSED:
            return True
        if state == self.HALF_OPEN and not self._trial_running:
            self._trial_running = True
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._trial_running = False
        self._set(self.CLOSED)

    def record_failure(self) -> None:
        self._trial_running = False
        self._failures += 1
        if self._state == self.HALF_OPEN or self._failures >= self.failure_threshold:
            self._opened_at = self._clock()
            if self._state != self.OPEN:
                logger.warning("circuit_open agent=%s failures=%s", self.name, self._failures)
            self._set(self.OPEN)

    def _set(self, state: str) -> None:
        self._state = state
        CIRCUIT_STATE.labels(self.name).set(self._GAUGE[state])


def is_connect_error(error: BaseException) -> bool:
    """True when the request never reached the server, so it is safe to send it again.

    The A2A client wraps httpx errors in its own exception types, so walk the cause chain.
    """
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout)):
            return True
        error = error.__cause__ or error.__context__
    return False


async def retry_on_connect_error(
    call: Callable[[], Awaitable[T]], *, name: str, retries: int, backoff: float,
) -> T:
    """Run `call`; on a connection error wait backoff, 2*backoff, ... and try again."""
    attempt = 0
    while True:
        try:
            return await call()
        except Exception as e:
            if attempt >= retries or not is_connect_error(e):
                raise
            wait = backoff * (2 ** attempt)
            attempt += 1
            CLIENT_RETRIES.labels(name).inc()
            logger.warning("a2a_retry agent=%s attempt=%s wait_s=%.1f error=%s", name, attempt, wait, e)
            await asyncio.sleep(wait)
