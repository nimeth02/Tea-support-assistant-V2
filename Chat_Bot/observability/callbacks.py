"""LangChain callback handler that records LLM, tool and graph-node spans for a trace."""
import logging
import threading
import time
from typing import Any, Dict, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from observability import metrics, store

logger = logging.getLogger("observability")

# Chains worth showing in a trace; everything else (prompt templates, parsers, ...) is noise
TRACKED_CHAINS = {
    "classify_intent", "disease_agent", "fertilize_agent",
    "help_agent", "maintenance_agent", "AgentExecutor",
}


class ObservabilityCallback(BaseCallbackHandler):
    raise_error = False

    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self._lock = threading.Lock()
        self._open: Dict[UUID, Dict[str, Any]] = {}
        self._parents: Dict[UUID, Optional[UUID]] = {}
        self._closed_ids: set = set()

    # ---- helpers -------------------------------------------------------
    def _recorded_parent(self, parent_run_id: Optional[UUID]) -> Optional[str]:
        """Walk up to the nearest ancestor that is itself recorded as a span."""
        while parent_run_id is not None:
            if parent_run_id in self._open or str(parent_run_id) in self._closed_ids:
                return str(parent_run_id)
            parent_run_id = self._parents.get(parent_run_id)
        return None

    def _start(self, run_id: UUID, parent_run_id: Optional[UUID], kind: str,
               name: str, input_: Any, model: Optional[str] = None) -> None:
        with self._lock:
            self._parents[run_id] = parent_run_id
            self._open[run_id] = {
                "span_id": str(run_id),
                "trace_id": self.trace_id,
                "parent_id": self._recorded_parent(parent_run_id),
                "kind": kind,
                "name": name,
                "model": model,
                "input": input_,
                "start_ts": time.time(),
                "_t0": time.perf_counter(),
            }

    def _end(self, run_id: UUID, status: str, output: Any = None,
             error: Optional[BaseException] = None, **extra: Any) -> Optional[Dict[str, Any]]:
        with self._lock:
            span = self._open.pop(run_id, None)
            if span is None:
                return None
            self._closed_ids.add(str(run_id))
        span["latency_ms"] = (time.perf_counter() - span.pop("_t0")) * 1000
        span["status"] = status
        span["output"] = output
        span["error"] = f"{type(error).__name__}: {error}" if error else None
        span.update(extra)
        try:
            store.save_span(span)
        except Exception:
            logger.exception("Failed to persist span")
        return span

    def _track_parent_only(self, run_id: UUID, parent_run_id: Optional[UUID]) -> None:
        with self._lock:
            self._parents[run_id] = parent_run_id

    # ---- LLM -----------------------------------------------------------
    @staticmethod
    def _model_name(serialized: Dict[str, Any], kwargs: Dict[str, Any]) -> str:
        params = kwargs.get("invocation_params") or {}
        return (params.get("model") or params.get("model_name")
                or (serialized or {}).get("kwargs", {}).get("model") or "unknown")

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        prompt = "\n".join(m.content if isinstance(m.content, str) else str(m.content)
                           for batch in messages for m in batch)
        self._start(run_id, parent_run_id, "llm", "chat_model", prompt,
                    model=self._model_name(serialized, kwargs))

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        self._start(run_id, parent_run_id, "llm", "llm", "\n".join(prompts),
                    model=self._model_name(serialized, kwargs))

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs):
        text, in_tok, out_tok = "", None, None
        try:
            gen = response.generations[0][0]
            text = gen.text
            usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
            in_tok, out_tok = usage.get("input_tokens"), usage.get("output_tokens")
        except (IndexError, AttributeError):
            pass
        span = self._end(run_id, "ok", text, input_tokens=in_tok, output_tokens=out_tok)
        if span:
            model = span["model"] or "unknown"
            metrics.LLM_CALLS.labels(model, "ok").inc()
            metrics.LLM_LATENCY.labels(model).observe(span["latency_ms"] / 1000)
            if in_tok:
                metrics.LLM_TOKENS.labels(model, "input").inc(in_tok)
            if out_tok:
                metrics.LLM_TOKENS.labels(model, "output").inc(out_tok)
            logger.info("llm_call model=%s latency_ms=%.0f tokens_in=%s tokens_out=%s",
                        model, span["latency_ms"], in_tok, out_tok)

    def on_llm_error(self, error, *, run_id, **kwargs):
        span = self._end(run_id, "error", error=error)
        if span:
            metrics.LLM_CALLS.labels(span["model"] or "unknown", "error").inc()
            logger.warning("llm_error model=%s error=%s", span["model"], error)

    # ---- Tools ---------------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        name = kwargs.get("name") or (serialized or {}).get("name") or "tool"
        self._start(run_id, parent_run_id, "tool", name, input_str)

    def on_tool_end(self, output, *, run_id, **kwargs):
        span = self._end(run_id, "ok", output)
        if span:
            metrics.TOOL_CALLS.labels(span["name"], "ok").inc()
            metrics.TOOL_LATENCY.labels(span["name"]).observe(span["latency_ms"] / 1000)
            logger.info("tool_call tool=%s latency_ms=%.0f", span["name"], span["latency_ms"])

    def on_tool_error(self, error, *, run_id, **kwargs):
        span = self._end(run_id, "error", error=error)
        if span:
            metrics.TOOL_CALLS.labels(span["name"], "error").inc()
            logger.warning("tool_error tool=%s error=%s", span["name"], error)

    # ---- Chains / graph nodes -----------------------------------------
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        name = kwargs.get("name") or (serialized or {}).get("name")
        if name in TRACKED_CHAINS:
            kind = "agent" if name == "AgentExecutor" else "node"
            self._start(run_id, parent_run_id, kind, name, inputs)
        else:
            self._track_parent_only(run_id, parent_run_id)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._end(run_id, "ok", outputs)

    def on_chain_error(self, error, *, run_id, **kwargs):
        self._end(run_id, "error", error=error)
