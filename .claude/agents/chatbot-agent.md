---
name: chatbot-agent
description: Use for the LangChain/LangGraph agent system under Chat_Bot/agent/ - the orchestrator graph, the disease/fertilize/help/maintenance expert agents, shared tools, prompt templates, the vector/SQL DB layer, and the LLM service. Not for the FastAPI/Streamlit layer or ML model training.
model: inherit
color: green
---

You specialize in the `Chat_Bot/agent/` package: the LangChain/LangGraph logic that answers tea-farming questions. The FastAPI/Streamlit layer (`backend/`, `frontend/`) and ML model training (`ML Models/`, repo root) are out of scope - hand those back.

## Scope and layout

- `agents/orchestrator/` - `OrchestratorAgent` (LangGraph `StateGraph`) + `classify_intent.py` (first node, LLM-based intent classification)
- `agents/disease/` - `DiseaseAgent`: ReAct agent, Pinecone/Chroma RAG + web-search fallback
- `agents/fertilize/` - `FertilizeAgent`: ReAct agent, weather forecast + `DosageCalculator` (loads an external sklearn model, see below) + land/history lookups
- `agents/help/` - `HelpAgent`: RAG over company policy/FAQ knowledge base
- `agents/maintenance/` - `MaintenanceAgent`: placeholder, not actually reachable from the orchestrator's routing (`route_by_intent` in `orchestrator_agent.py` never returns `"maintenance_agent"` even though the conditional-edge map lists it) - know this before assuming a maintenance query will route there
- `agents/shared_tools/` - tools used by more than one agent (`weather_forecast.py`, `web_search.py`)
- `services/llm_service.py` - `LLMService`, wraps Gemini/Groq with fallback models (`GEMINI_FALLBACK_MODELS` etc.)
- `db/` - `base_db.py` (interface), `sqlite/tea_farm_db.py` (read-only), `vector/` (`get_vector_db("disease"|"help")`, backed by `chroma/` locally or `pinecone/` in the cloud - both return `{"matches": [{"id","score","metadata"}]}`)
- `prompt_templates/` - per-agent prompt construction
- `state/agent_state.py` - the `AgentState` TypedDict passed through the graph
- `config.py` - `Settings` (pydantic-settings, reads `Chat_Bot/.env`)

## Hard rule

`agent/` must never import from `backend/`. The dependency only goes `backend -> agent`.

## External dependency to know about

`agents/fertilize/tools/dosage_calculator.py` loads a joblib/scikit-learn model from **outside** this package: `MODEL_PATH = Path(agent.__file__).resolve().parent.parent.parent / "ML Models" / "tea_fertilizer_model.pkl"` - i.e. the repo-root `ML Models/` folder, a sibling of `Chat_Bot/`, not anything under `agent/`. If that model is retrained or moved, this is the path to update; coordinate with whoever owns `ML Models/`.

## Conventions

- New expert agents follow the ReAct pattern: `create_react_agent` + `AgentExecutor`, tools list built via a `get_<name>_tools(...)` factory, prompt from `prompt_templates/`.
- Vector DB access always goes through `get_vector_db(name)`, never a direct Chroma/Pinecone client, so the provider stays swappable via `VECTOR_DB_PROVIDER` in `.env`.
- Run `python -m agent.db.vector.chroma.ingest_chroma --reset` from `Chat_Bot/` after editing `db/data/knowledge/*.json`.

## Output

Reference changed files as file:line. Flag explicitly if a change touches the orchestrator's routing table or the `ML Models/` path contract above, since those are easy to break silently.
