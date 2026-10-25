"""AIOps runtime layer: what keeps the agents observable and safe once they run in production.

  telemetry.py   trace id that follows one farmer question across every agent, Prometheus metrics
  log_config.py  structured (JSON) logs that carry the trace id and the agent's name
  resilience.py  retries and circuit breakers, so one broken expert can't take the whole chat down
  guardrails.py  input checks (size, prompt injection) and output clean-up (paths, secrets)
  health.py      liveness / readiness checks used by Docker, Kubernetes and the smoke test

Nothing here depends on a specific agent framework, so CrewAI, LangChain and LangGraph experts
all share it.
"""
