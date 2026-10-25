"""Trace context and Prometheus metrics shared by the orchestrator and every expert.

Trace id: one short random id per farmer question. The backend creates it, the orchestrator
puts it into every A2A message's metadata, and each expert puts it into its own logs. So when
something goes wrong you can search all services' logs for one id and see the whole story.

Metrics: numbers Prometheus collects every few seconds (counts, latencies, states). Grafana
draws them and the alert rules in deploy/prometheus/alert_rules.yml fire on them.
All names start with "a2a_" or "guardrail_" so they never clash with the backend's own metrics
(observability/metrics.py), which live in the same Prometheus registry when the backend imports
the orchestrator.
"""
import uuid
from contextvars import ContextVar

from prometheus_client import Counter, Gauge, Histogram

_trace_id: ContextVar[str] = ContextVar("a2a_trace_id", default="-")
_agent_name: ContextVar[str] = ContextVar("a2a_agent_name", default="-")


def set_trace_id(trace_id: str | None) -> str:
    """Use the given trace id for this request (or make a new one) and return it."""
    trace_id = trace_id or uuid.uuid4().hex[:16]
    _trace_id.set(trace_id)
    return trace_id


def get_trace_id() -> str:
    return _trace_id.get()


def set_agent_name(name: str) -> None:
    _agent_name.set(name)


def get_agent_name() -> str:
    return _agent_name.get()


LATENCY_BUCKETS = (0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 90, 120)

# ---- Expert side (one set per expert server) -----------------------------------------------
EXPERT_REQUESTS = Counter(
    "a2a_expert_requests_total",
    "A2A questions answered by an expert",
    ["agent", "framework", "status"],  # status: ok | error | timeout | rejected
)
EXPERT_LATENCY = Histogram(
    "a2a_expert_latency_seconds",
    "Time an expert spent answering one question",
    ["agent", "framework"],
    buckets=LATENCY_BUCKETS,
)
EXPERT_TOOL_CALLS = Counter(
    "a2a_expert_tool_calls_total", "Tools an expert used", ["agent", "tool"]
)
EXPERT_LLM_FALLBACKS = Counter(
    "a2a_expert_llm_fallbacks_total",
    "Times an expert had to switch to a fallback LLM",
    ["agent", "model"],
)
EXPERT_IN_FLIGHT = Gauge(
    "a2a_expert_in_flight", "Questions an expert is working on right now", ["agent"]
)

# ---- Orchestrator side (calls out to the experts) ------------------------------------------
CLIENT_CALLS = Counter(
    "a2a_client_calls_total",
    "A2A calls from the orchestrator to an expert",
    ["agent", "status"],  # status: ok | error | timeout | circuit_open
)
CLIENT_LATENCY = Histogram(
    "a2a_client_latency_seconds",
    "Round-trip time of one A2A call, as seen by the orchestrator",
    ["agent"],
    buckets=LATENCY_BUCKETS,
)
CLIENT_RETRIES = Counter(
    "a2a_client_retries_total", "Connection retries to an expert", ["agent"]
)
CIRCUIT_STATE = Gauge(
    "a2a_circuit_state",
    "Circuit breaker per expert: 0 closed (normal), 1 half-open (testing), 2 open (blocked)",
    ["agent"],
)
AGENTS_DISCOVERED = Gauge(
    "a2a_agents_discovered", "Experts whose Agent Card the orchestrator could read"
)
ORCHESTRATOR_TURNS = Counter(
    "a2a_orchestrator_turns_total",
    "Farmer messages handled by the orchestrator",
    ["status"],  # ok | error | blocked | unavailable
)

# ---- Guardrails ----------------------------------------------------------------------------
GUARDRAIL_EVENTS = Counter(
    "guardrail_events_total",
    "Inputs blocked or outputs cleaned by a guardrail",
    ["stage", "reason"],  # stage: input | output
)
