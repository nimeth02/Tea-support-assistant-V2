"""Structured logging for the expert servers.

Production log tools (Loki, ELK, CloudWatch, Datadog) search logs by field. So each log line is
one JSON object, for example:

    {"ts": "2026-10-24T10:31:02Z", "level": "INFO", "service": "disease-agent",
     "trace_id": "9f2c1d0a7b3e4f11", "logger": "agent.a2a", "msg": "expert_answer ...", ...}

Set LOG_FORMAT=text for human-friendly lines while developing.
"""
import json
import logging
import sys
from datetime import datetime, timezone

from agent.config import settings
from agent.ops.telemetry import get_agent_name, get_trace_id

# Extra keys passed with logger.info(..., extra={...}) that should appear in the JSON.
_EXTRA_FIELDS = ("agent", "framework", "status", "latency_ms", "tools", "user_id", "remote_agent")


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = get_trace_id()
        record.service = get_agent_name()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname,
            "service": getattr(record, "service", "-"),
            "trace_id": getattr(record, "trace_id", "-"),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in _EXTRA_FIELDS:
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(service: str) -> None:
    """Send this process's logs to stdout (where Docker collects them) in the chosen format."""
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_ContextFilter())
    if settings.LOG_FORMAT.lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(service)s trace=%(trace_id)s] %(name)s: %(message)s"
        ))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.LOG_LEVEL.upper())
    # Very chatty libraries: keep only their warnings.
    for noisy in ("httpx", "httpcore", "LiteLLM", "litellm", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
