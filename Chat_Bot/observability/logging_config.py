import logging
import sys

from observability.context import get_trace_id


class TraceIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = get_trace_id()
        return True


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(TraceIdFilter())
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [trace=%(trace_id)s] %(name)s: %(message)s"
    ))
    for name in ("observability", "backend", "agent"):
        log = logging.getLogger(name)
        log.handlers = [handler]
        log.setLevel(level)
        log.propagate = False
