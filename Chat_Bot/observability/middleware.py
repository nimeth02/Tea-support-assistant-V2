"""HTTP middleware that assigns a trace id to every request and records HTTP metrics/logs."""
import logging
import time

from fastapi import FastAPI, Request

from observability import metrics
from observability.context import new_trace_id

logger = logging.getLogger("observability")

# Paths excluded from HTTP metrics/logging to avoid noise from scraping and dashboard polling
UNTRACKED_PREFIXES = ("/metrics", "/observability")


def add_observability_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def observability_middleware(request: Request, call_next):
        trace_id = request.headers.get("X-Trace-Id") or new_trace_id()
        request.state.trace_id = trace_id
        path = request.url.path
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Trace-Id"] = trace_id
            return response
        finally:
            elapsed = time.perf_counter() - start
            if not path.startswith(UNTRACKED_PREFIXES):
                metrics.HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
                metrics.HTTP_LATENCY.labels(request.method, path).observe(elapsed)
                logger.info("http_request method=%s path=%s status=%s latency_ms=%.0f",
                            request.method, path, status, elapsed * 1000)
