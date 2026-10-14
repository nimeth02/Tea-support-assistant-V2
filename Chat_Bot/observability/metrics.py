from prometheus_client import Counter, Histogram

LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120)

HTTP_REQUESTS = Counter(
    "http_requests_total", "HTTP requests", ["method", "path", "status"]
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "path"],
    buckets=LATENCY_BUCKETS,
)

CHAT_REQUESTS = Counter("chat_requests_total", "Chat requests by intent", ["intent", "status"])
CHAT_LATENCY = Histogram(
    "chat_duration_seconds", "End-to-end chat latency", ["intent"], buckets=LATENCY_BUCKETS
)

LLM_CALLS = Counter("llm_calls_total", "LLM calls", ["model", "status"])
LLM_LATENCY = Histogram(
    "llm_call_duration_seconds", "LLM call latency", ["model"], buckets=LATENCY_BUCKETS
)
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens", ["model", "type"])

TOOL_CALLS = Counter("tool_calls_total", "Agent tool calls", ["tool", "status"])
TOOL_LATENCY = Histogram(
    "tool_call_duration_seconds", "Agent tool latency", ["tool"], buckets=LATENCY_BUCKETS
)
