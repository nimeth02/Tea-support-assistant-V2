# Multi-framework A2A agents (variant key: `hybrid`)

Three expert agents, each built with a **different agent framework**, working together through
the **A2A protocol** (Agent2Agent, an open standard for agents to talk over HTTP). A LangChain
supervisor decides which experts to ask and writes one reply for the farmer.

```
                      farmer (Streamlit UI :8514)
                                 |
                      backend :8005  (FastAPI)
                                 |
           LangChain supervisor  +  guardrails  +  A2A client
           (retries, circuit breakers, trace id, auth token)
              |                      |                      |
         A2A / HTTP             A2A / HTTP             A2A / HTTP
              |                      |                      |
   Disease expert :9201    Fertilizer expert :9202   Help desk :9203
        CrewAI                  LangChain                LangGraph
   RAG + leaf-photo CNN    farm DB + dosage model    retrieve -> answer graph
   + web search            + weather forecast
```

## Why three frameworks?

Each framework is good at a different shape of problem. A2A lets each expert use the best fit,
because the experts only meet at the protocol. The supervisor reads each expert's **Agent
Card** (a small JSON file describing it) and never sees which framework is inside.

| Expert | Framework | Why this framework fits | Tools |
|---|---|---|---|
| Disease | **CrewAI** | Role/goal/backstory style suits an "expert persona" that reasons across several sources | TeaDiseaseRAG, ClassifyLeafImage (ResNet18), DuckDuckGoWebSearch |
| Fertilizer | **LangChain** `create_agent` | Native tool calling over structured farmer data | TeaLandDataByUser, GetFertilizeHistoryByUser, DosageCalculator (sklearn), WeatherForecast |
| Company help | **LangGraph** `StateGraph` | A fixed, auditable path: it may only answer from the help database | CompanyHelpRAG, then one LLM call (or a fixed "not found" reply with no LLM call) |
| Orchestrator | **LangChain** `create_agent` | Each expert becomes one tool, built from its Agent Card | one `ask_<expert>` tool per card |

Swapping an expert's framework changes nothing outside it. Adding an expert means starting
its server and adding its URL. No supervisor code or prompt changes.

## How one question flows

1. The backend gets the message (and maybe a leaf photo) and gives it a **trace id**, a short
   random id used to follow this one question through every service.
2. **Guardrails** check the text: empty, too long, or a prompt-injection attempt ("ignore all
   previous instructions...") is answered right away, with no LLM call.
3. The supervisor LLM picks the experts. For each one it sends an A2A message carrying the
   question, the farmer's id and trace id in the metadata, the photo bytes (only to experts
   whose card accepts images), and the shared auth token.
4. Each expert checks the token, validates the input, answers within a time limit, and replies
   with the answer plus metadata: framework, tools used, latency, and whether it was a fallback.
5. The supervisor combines the answers, guardrails remove file paths and secret-looking strings,
   and the backend returns the reply.

## Run it locally

From the `Chat_Bot` folder (with `GEMINI_API_KEY` in `Chat_Bot/.env`):

```
python launch.py up hybrid          # 3 experts + backend :8005 + UI http://localhost:8514
```

Or piece by piece:

```
python "agent (crew AI , langchain)/run_hybrid.py" experts          # the 3 experts, one process each
python "agent (crew AI , langchain)/run_hybrid.py" smoke --ask      # are they up, ready, answering?
python "agent (crew AI , langchain)/run_hybrid.py" eval             # golden-set quality gate
python -m pytest "agent (crew AI , langchain)/tests" -q             # 39 tests, no LLM calls
```

The experts use ports 9201-9203, so this variant can run next to the `a2a` variant (9101-9103).

## The AIOps layer: what makes it production-ready

AIOps here means the practices that keep an AI system observable, reliable and safe while it runs.

| Area | What it does | Where |
|---|---|---|
| **Health checks** | `/healthz` answers "is the process alive?" and `/readyz` answers "can it work?" (LLM key, model file, database, vector store). Docker restarts on the first and holds traffic on the second. | `ops/health.py` |
| **Metrics** | Prometheus counters and histograms: requests and latency per expert and framework, tools used, LLM fallbacks, in-flight questions, A2A calls, retries, circuit state, guardrail events | `ops/telemetry.py`, `/metrics` on every service |
| **Structured logs** | One JSON object per line, each with `trace_id`, `service` and `latency_ms`, so log tools can search them by field | `ops/log_config.py` |
| **Distributed tracing** | The same trace id in every expert's logs. With LangSmith on, CrewAI, LangChain and LangGraph steps all nest in one trace. | `a2a_client.py`, `a2a_servers/common.py` |
| **Resilience** | Timeouts. Retries only when the request never arrived, so an LLM never runs twice. A circuit breaker per expert: after 3 failures in a row it fails fast for 30s, then tests once. Experts that were down are found again automatically. Model fallbacks: Gemini models, then Groq. | `ops/resilience.py`, `services/llm_service.py` |
| **Graceful degradation** | If an expert is down, the farmer still gets the other answers, plus a note that one part can't be answered right now | `UNAVAILABLE_NOTE`, orchestrator prompt |
| **Security** | A shared bearer token between agents, advertised in the Agent Card and compared in constant time. Production refuses to start without it. Containers run as a non-root user. | `a2a_servers/common.py`, `deploy/Dockerfile` |
| **Guardrails** | Input checks (size, control characters, prompt injection) and output clean-up (paths, keys, length) | `ops/guardrails.py` |
| **Evaluation** | 19 golden questions covering single-expert, mixed, off-topic and injection cases. Scores routing accuracy, keyword hits, error rate and p50/p95 latency. Logs to MLflow when `MLFLOW_TRACKING_URI` is set. Fails below 80% routing accuracy. | `evals/` |
| **Testing** | A2A contract tests with real servers in memory and fake experts: discovery, metadata both ways, auth, photo forwarding, timeouts, circuit breaker, rediscovery | `tests/` |
| **Deployment** | One image, one container per expert, healthchecks, memory limits, restart policy, Prometheus, alert rules, Alertmanager, Grafana dashboard | `deploy/` |
| **Alerting** | Alerts for an expert or the backend going down, an open circuit, a high error rate, slow answers, wrong-token calls, injection spikes and the chat failure rate. Each carries a short runbook (what to check first). | `deploy/prometheus/alert_rules.yml` |
| **CI/CD** | Tests on every push, an image build on main, and a nightly smoke test plus eval gate on the real stack | `deploy/ci/github-actions.yml` |

### Deploy with Docker

```
cd "Chat_Bot/agent (crew AI , langchain)/deploy"
cp .env.example .env            # set GEMINI_API_KEY and A2A_AUTH_TOKEN
docker compose up -d --build
```

UI :8514, backend :8005, Prometheus :9090, Alertmanager :9093, Grafana :3000 (dashboard "Tea agents (A2A)").

To turn on CI, copy `deploy/ci/github-actions.yml` to `.github/workflows/` at the repo root and
add the `GEMINI_API_KEY` and `A2A_AUTH_TOKEN` repository secrets.

## Settings added by this variant

All have defaults (see `config.py`). The ones you will most likely touch:

| Setting | Default | Meaning |
|---|---|---|
| `A2A_AUTH_TOKEN` | empty | Shared secret between agents. Required when `ENVIRONMENT=production` |
| `ENVIRONMENT` | `dev` | `production`/`staging` turn on strict checks |
| `A2A_*_AGENT_URL` | `localhost:9201-9203` | Where each expert runs |
| `CIRCUIT_FAILURE_THRESHOLD`, `CIRCUIT_RESET_SECONDS` | 3, 30 | Circuit breaker |
| `EXPERT_TIMEOUT_SECONDS`, `A2A_TIMEOUT_SECONDS` | 90, 120 | Time limits per expert answer and per call |
| `LOG_FORMAT` | `json` | `text` is easier to read by eye |
| `FERTILIZER_MODEL_PATH`, `DISEASE_MODEL_PATH` | `../ML Models/...` | Model files, if mounted elsewhere |

## Adding a new expert

1. Write the expert in any framework, with `name`, `framework` and an async `answer()` that
   returns an `ExpertAnswer` (see `agents/base.py`).
2. Add `a2a_servers/<name>_server.py` with its Agent Card and readiness checks. Copy
   `help_server.py`.
3. Give it a port in `config.py`, add it to `run_agents.py`, and add it to `docker-compose.yml`
   and `prometheus.yml`.

The supervisor picks it up from its card automatically.

## Known limits

- The maintenance agent is still a placeholder and is not served.
- A2A tasks are kept in memory, which is fine for one replica per expert. Running several
  replicas behind a load balancer would need a database task store.
- The Gemini rate limiter is per process, so three experts plus the orchestrator can together
  go over the free-tier per-minute limit. The fallback models absorb most of that.
