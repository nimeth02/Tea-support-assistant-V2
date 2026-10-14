## Layout

```
backend/        FastAPI layer only: main.py, routes, request/response models, dependencies
agent/          LangChain/LangGraph agent
  agents/         One folder per agent:
    orchestrator/   OrchestratorAgent (LangGraph graph) + ClassifyIntent (first node)
    disease/        DiseaseAgent   (ReAct: Pinecone RAG + web search)
    fertilize/      FertilizeAgent (ReAct: weather, dosage model, land + history data)
    help/           HelpAgent      (Pinecone RAG over company help/policies)
    maintenance/    MaintenanceAgent (placeholder, not wired up yet)
    shared_tools/   Tools used by more than one agent (weather_forecast.py, web_search.py)
  services/       Shared infrastructure for the agents (llm_service.py)
  db/
    base_db.py      shared abstract interface
    sqlite/         SQLite access code (tea_farm_db.py, read-only connection)
    vector/         vector DB clients behind get_vector_db("disease"|"help"):
                      base_vector_db.py  common VectorDatabase interface
                      chroma/            ChromaDatabase + ingest_chroma.py (local, on disk; built from data/knowledge/*.json)
                      pinecone/          disease_vector_db.py, help_vector_db.py (cloud, Cohere embeddings)
    data/           data files only:
                      tea_farm.db           SQLite: tea land + fertilization history (read-only)
                      knowledge/*.json      RAG knowledge bases (tea_disease.json, company_help.json)
                      chroma/               generated Chroma store (gitignored, rebuild with ingest)
  prompt_templates/, state/, config.py (.env settings)
observability/  Tracing, metrics, logging, SQLite trace store, /metrics + /observability routes
frontend/       Streamlit UI (chat + observability dashboard)
data/           Runtime data (observability.db) - gitignored

../ML Models/   (sibling of Chat_Bot/, repo root) Tea Fertilizer Dosage/model/tea_fertilizer_model.pkl,
                  Tea Disease Classification/model/tea_disease_resnet18.pkl (not yet wired to DiseaseAgent),
                  mlflow/ (shared MLflow tracking store for both)
```

Dependency direction: `backend` -> `agent`, `backend` -> `observability`. The `agent` package does not import from `backend`.

## Run

Run all commands from this folder (`Chat_Bot`), so that `backend`, `agent` and `observability` are importable and `.env` is found.

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env        # then set GEMINI_API_KEY

# Build the local vector knowledge bases (once, and whenever data/knowledge/*.json changes)
.venv\Scripts\python -m agent.db.vector.chroma.ingest_chroma --reset

# Backend (FastAPI)
.venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001

# Frontend (Streamlit) - set TEA_API_URL if the backend runs elsewhere
.venv\Scripts\python -m streamlit run frontend\streamlit_app.py --server.port 8510
```

## Vector DB (RAG)

`VECTOR_DB_PROVIDER` in `.env` selects the backend; agents only ever call `get_vector_db("disease")` / `get_vector_db("help")`.

- `chroma` (default) - local Chroma store in `agent/db/data/chroma/`, embeddings computed on your machine
  (all-MiniLM-L6-v2, ~80 MB downloaded once). No API keys. Knowledge comes from `agent/db/data/knowledge/*.json`;
  edit those files and re-run the ingest command. The shipped records are a small sample set.
- `pinecone` - the original cloud indexes with Cohere embeddings. Needs `PINECONE_API_KEY` and `COHERE_API_KEY`.

Both return the same shape (`{"matches": [{"id", "score", "metadata"}]}`) so the RAG tools and agents are unchanged.

## Observability

Everything lives in the `observability/` package:

- `context.py` - per-request trace id (contextvar)
- `middleware.py` - FastAPI middleware: assigns `X-Trace-Id`, records HTTP metrics and logs
- `callbacks.py` - LangChain callback that records LLM / tool / graph-node spans
- `metrics.py` - Prometheus counters and histograms
- `store.py` - SQLite store for traces and spans
- `logging_config.py` - log format with `[trace=<id>]`
- `routes.py` - `GET /metrics` and `GET /observability/*` endpoints

Behaviour:

- Every chat request gets a trace id (returned in the `X-Trace-Id` header and the response body).
- LLM calls, agent tool calls and LangGraph nodes are recorded as spans (latency, tokens, model, errors) in `data/observability.db` (SQLite).
- Streamlit **Observability** page: KPIs, latency, intents, tool/LLM usage, and a per-trace span waterfall.
- `GET /metrics` - Prometheus metrics (HTTP, chat, LLM, tokens, tools).
- `GET /observability/stats`, `/observability/traces`, `/observability/traces/{id}`, `/observability/spans` - JSON APIs.
- Logs include `[trace=<id>]` so they can be correlated with traces.

### LangSmith (optional)

Set in `.env` to turn on: `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY=<your key>` (free at
https://smith.langchain.com), `LANGSMITH_PROJECT=tea-support-assistant`.

`agent/config.py` mirrors these into `os.environ` at import time (LangChain reads them straight
from the environment, not from the `Settings` object), so no other code change is needed -
every LLM call, tool call and LangGraph node then shows up automatically in the LangSmith UI as
a nested trace, with no need to pass the existing `ObservabilityCallback` any differently. This
runs *alongside* the custom Prometheus/SQLite observability above, not instead of it: LangSmith
is for inspecting/debugging individual agent runs, Prometheus/SQLite is for ops-level metrics and
the Streamlit dashboard. Leave `LANGSMITH_TRACING` unset/false to skip it entirely.





🚀 Tea Farming Support Assistant – AI-Powered Farming Help 🌱

I’ve been working on an AI-driven Tea Farming Support Assistant – a modular chatbot designed to answer farmers’ questions on tea cultivation.

💡 How it works:
A central OrchestratorAgent (LangGraph, served through FastAPI) classifies a farmer’s query using an LLM (Groq API) and routes it to the right expert module:

🛡 DiseaseAgent – Tea Disease Advisor

Tools:

TeaDiseaseRAG – Retrieves local disease & pest info from Pinecone vector DB

DuckDuckGoWebSearch – Falls back on web search for broader info

Purpose: Combine local knowledge + global sources for accurate disease diagnosis.

🌾 FertilizeAgent – Fertilizer & Soil Advisor

Tools:

WeatherForecast – 14-day rainfall & precipitation prediction

DosageCalculator – Mock ML model (scikit-learn/joblib) for optimal fertilizer dosage

TeaLandDataByUser – Retrieves farmer’s land size, location, soil type

GetFertilizeHistoryByUser – Fetches past fertilization records

Purpose: Provide personalized fertilizer plans based on land, weather, and history.

📄 HelpAgent – Company Policy & Support

Tools:

HelpRAG – Retrieves company policies & FAQ answers from Pinecone vector DB

Purpose: Answer queries about tea collection dates, loans, advances, and policies.

⚙ Key Technologies:

Backend: FastAPI (async, modular design)

Frontend: Next.js (interactive UI)

AI Frameworks: LangChain, LangGraph

LLM: Groq API

Vector Search: Pinecone

External APIs: Weather forecast integration

Mock ML Model: Fertilizer dosage prediction (scikit-learn/joblib)