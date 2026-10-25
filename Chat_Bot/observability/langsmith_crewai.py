"""Send CrewAI's steps to LangSmith.

Why this exists: LangChain-based agents are traced to LangSmith automatically, but CrewAI is a
different framework and does not talk to LangSmith. CrewAI does announce everything it does on
its *event bus* ("crew started", "LLM call finished", "tool used", ...). This listener turns
those events into LangSmith runs:

  - a "...started" event opens a run, nested under the run of its parent event;
  - the matching "...finished/failed" event (CrewAI links them via started_event_id) closes it.

CrewAI doesn't always link LLM/tool events to their agent (agents may run in another thread),
so those fall back to the open run of the agent with the same role.

A top-level event (the crew kickoff) is nested under the backend's chat run. CrewAI hands each
event to listeners with a copy of the caller's context, so that run is still "current" here.

Call install() once per process, before any crew runs.
"""
import logging
import threading
from typing import Any, Dict, Optional

from crewai.events import BaseEventListener
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent, AgentExecutionErrorEvent, AgentExecutionStartedEvent,
)
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent, CrewKickoffFailedEvent, CrewKickoffStartedEvent,
)
from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallFailedEvent, LLMCallStartedEvent
from crewai.events.types.task_events import TaskCompletedEvent, TaskFailedEvent, TaskStartedEvent
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent, ToolUsageFinishedEvent, ToolUsageStartedEvent,
)
from langsmith.run_helpers import get_current_run_tree
from langsmith.run_trees import RunTree

from observability.langsmith_tracing import tracing_enabled

logger = logging.getLogger("observability")


def _text(value: Any, limit: int = 20000) -> str:
    text = value if isinstance(value, str) else str(value)
    return text[:limit]


def _short(value: str, limit: int = 60) -> str:
    """A run name: first line only, cut to `limit` characters."""
    line = next((ln.strip() for ln in (value or "").splitlines() if ln.strip()), "")
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _role(event) -> Optional[str]:
    return event.agent_role or getattr(getattr(event, "from_agent", None), "role", None)


class LangSmithCrewListener(BaseEventListener):
    def __init__(self):
        self._runs: Dict[str, RunTree] = {}  # started event id -> open LangSmith run
        self._agent_runs: Dict[str, RunTree] = {}  # agent role -> its open agent run
        self._lock = threading.Lock()
        super().__init__()

    # ---- helpers -------------------------------------------------------
    def _open(self, event, name: str, run_type: str, inputs: Dict[str, Any],
              extra: Optional[Dict[str, Any]] = None) -> Optional[RunTree]:
        if not tracing_enabled():
            return None
        try:
            with self._lock:
                parent = self._runs.get(event.parent_event_id)
                if parent is None and run_type in ("llm", "tool"):
                    parent = self._agent_runs.get(_role(event))
            parent = parent or get_current_run_tree()
            if parent is not None:
                run = parent.create_child(name=name, run_type=run_type, inputs=inputs,
                                          start_time=event.timestamp, extra=extra)
            else:  # no backend run (e.g. a crew started from a script) - start a new trace
                run = RunTree(name=name, run_type=run_type, inputs=inputs,
                              start_time=event.timestamp, extra=extra or {})  # RunTree rejects None
            run.post()
            with self._lock:
                self._runs[event.event_id] = run
            return run
        except Exception:
            logger.exception("LangSmith: could not open run for %s", name)

    def _close(self, event, outputs: Optional[Dict[str, Any]] = None, error: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            run = self._runs.pop(event.started_event_id, None)
        if run is None:
            return
        try:
            run.end(outputs=outputs, error=error, end_time=event.timestamp, metadata=metadata)
            run.patch()
        except Exception:
            logger.exception("LangSmith: could not close run %s", run.name)

    # ---- event handlers ------------------------------------------------
    def setup_listeners(self, bus) -> None:
        @bus.on(CrewKickoffStartedEvent)
        def _(source, event):
            self._open(event, f"crew:{event.crew_name or 'crew'}", "chain", {"inputs": event.inputs or {}})

        @bus.on(CrewKickoffCompletedEvent)
        def _(source, event):
            self._close(event, {"output": _text(getattr(event.output, "raw", event.output))},
                        metadata={"total_tokens": event.total_tokens})

        @bus.on(CrewKickoffFailedEvent)
        def _(source, event):
            self._close(event, error=event.error)

        @bus.on(TaskStartedEvent)
        def _(source, event):
            description = getattr(event.task, "description", None) or event.context or ""
            name = getattr(event.task, "name", None) or _short(description)
            self._open(event, f"task:{name or 'task'}", "chain", {"description": _text(description)})

        @bus.on(TaskCompletedEvent)
        def _(source, event):
            self._close(event, {"output": _text(getattr(event.output, "raw", event.output))})

        @bus.on(TaskFailedEvent)
        def _(source, event):
            self._close(event, error=_text(getattr(event, "error", "task failed")))

        @bus.on(AgentExecutionStartedEvent)
        def _(source, event):
            role = getattr(event.agent, "role", None) or event.agent_role or "agent"
            run = self._open(event, f"agent:{role}", "chain", {"task_prompt": _text(event.task_prompt)})
            if run is not None:
                with self._lock:
                    self._agent_runs[role] = run

        def _agent_done(event):
            role = getattr(event.agent, "role", None) or event.agent_role
            with self._lock:
                self._agent_runs.pop(role, None)

        @bus.on(AgentExecutionCompletedEvent)
        def _(source, event):
            _agent_done(event)
            self._close(event, {"output": _text(event.output)})

        @bus.on(AgentExecutionErrorEvent)
        def _(source, event):
            _agent_done(event)
            self._close(event, error=event.error)

        @bus.on(LLMCallStartedEvent)
        def _(source, event):
            messages = event.messages if isinstance(event.messages, list) else [
                {"role": "user", "content": _text(event.messages or "")}]
            self._open(event, event.model or "llm", "llm", {"messages": messages},
                       extra={"invocation_params": {"model": event.model}, "metadata": {
                           "ls_model_name": event.model, "agent_role": event.agent_role}})

        @bus.on(LLMCallCompletedEvent)
        def _(source, event):
            usage = event.usage or {}
            in_tok = usage.get("prompt_tokens") or usage.get("input_tokens")
            out_tok = usage.get("completion_tokens") or usage.get("output_tokens")
            outputs: Dict[str, Any] = {"output": _text(event.response)}
            if in_tok is not None or out_tok is not None:
                # LangSmith reads token counts (and so cost) from this shape
                outputs["usage_metadata"] = {"input_tokens": in_tok or 0, "output_tokens": out_tok or 0,
                                             "total_tokens": (in_tok or 0) + (out_tok or 0)}
            self._close(event, outputs)

        @bus.on(LLMCallFailedEvent)
        def _(source, event):
            self._close(event, error=event.error)

        @bus.on(ToolUsageStartedEvent)
        def _(source, event):
            args = event.tool_args if isinstance(event.tool_args, dict) else {"input": event.tool_args}
            self._open(event, event.tool_name, "tool", args)

        @bus.on(ToolUsageFinishedEvent)
        def _(source, event):
            self._close(event, {"output": _text(event.output)}, metadata={"from_cache": event.from_cache})

        @bus.on(ToolUsageErrorEvent)
        def _(source, event):
            self._close(event, error=_text(event.error))


_listener: Optional[LangSmithCrewListener] = None


def install() -> None:
    """Register the listener on CrewAI's event bus (once per process)."""
    global _listener
    if _listener is None:
        _listener = LangSmithCrewListener()
