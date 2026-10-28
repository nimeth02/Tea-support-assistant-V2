# Getting started: how to run every project in this repo

This guide shows how to start each project in this repository, one by one, from a fresh
checkout. Every command is for **Windows** (PowerShell or Command Prompt). Run each one from the
folder named in its heading.

You don't need to activate virtual environments. Every command calls the environment's own
Python directly (for example `.venv\Scripts\python`), so the right packages are always used.
Where PowerShell needs different syntax from Command Prompt, both are shown.

---

## 1. What is in this repo

There are two groups of projects. The ML models are trained first, and the chatbot then uses
the trained model files.

| # | Project | What it is | Folder | Opens on |
|---|---|---|---|---|
| A1 | Tea Fertilizer Dosage | Trains the model that predicts how many kg of fertilizer to use | `ML Models/Tea Fertilizer Dosage/` | Jupyter notebook |
| A2 | Tea Disease Classification | Trains the model that recognises a disease from a leaf photo | `ML Models/Tea Disease Classification/` | Jupyter notebook |
| A3 | MLflow UI | A web page showing every training run and model version | `ML Models/mlflow/` | http://127.0.0.1:5000 |
| B1 | Chatbot: `agent` | LangGraph router: classify the question, then send it to one expert | `Chat_Bot/agent/` | http://localhost:8510 |
| B2 | Chatbot: `langchain` | A LangChain supervisor that calls the experts as tools | `Chat_Bot/agent (langchain)/` | http://localhost:8511 |
| B3 | Chatbot: `a2a` | The experts are separate web servers, reached over the A2A protocol | `Chat_Bot/agent A2A/` | http://localhost:8512 |
| B4 | Chatbot: `crewai` | A CrewAI manager that delegates to a crew of experts | `Chat_Bot/agent (crewAI)/` | http://localhost:8513 |
| B5 | Chatbot: `hybrid` | Each expert uses a different framework (CrewAI, LangChain, LangGraph), connected over A2A, with production monitoring | `Chat_Bot/agent (crew AI , langchain)/` | http://localhost:8514 |
| B6 | Chatbot: `voice` | The B1 router, plus Sinhala/Tamil/English translation and voice in and out | `Chat_Bot/agent multilangual and voice also/` | http://localhost:8515 |

All six chatbot versions answer the same farmer questions. They differ in how the agents are
built and connected (B1-B5) or in how the farmer can talk to them (B6), so you can compare them
side by side.

Some terms used below:
- **Virtual environment (`.venv`)**: a private folder of Python packages for one project, so
  projects don't break each other's packages.
- **Variant**: one of the six chatbot versions (B1-B6).
- **Backend**: the FastAPI server that runs the agents. **UI**: the Streamlit web page you chat in.
- **A2A** (Agent2Agent): an open protocol that lets agents talk to each other over HTTP, even
  when they are built with different frameworks.

---

## 2. One-time prerequisites

1. **Python 3.13** from python.org. Tick "Add python.exe to PATH" during install. Check it:
   ```
   python --version
   ```
2. **Git**, to clone the repo.
3. **API keys.** Only the first one is required to start.
   - `GEMINI_API_KEY` (required for the chatbot): free at https://aistudio.google.com/apikey
   - `WHETHER_API` (optional, for rain forecasts): free at https://www.weatherapi.com
   - `LANGSMITH_API_KEY` (optional, for viewing agent traces): free at https://smith.langchain.com
   - `GROQ_API_KEY` (optional, a backup LLM): free at https://console.groq.com
4. **Docker Desktop**, only if you want to run B5 in containers (section 3.5).

The trained model files are already in `ML Models/*/model/`. You can therefore skip Part A and
go straight to the chatbot (Part B). Do Part A when you want to retrain or inspect the models.

---

## Part A - ML models

### A1. Tea Fertilizer Dosage

Folder: `ML Models\Tea Fertilizer Dosage`

**First time only:**
```
cd "ML Models\Tea Fertilizer Dosage"
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt jupyter
.venv\Scripts\python dataset\generate_dataset.py
```
The last command creates the training data in `dataset\data\`. This data is synthetic: it is
generated from a formula, because no real historical data exists yet.

**Train:**
```
.venv\Scripts\python -m jupyter lab train_tea_fertilizer_model.ipynb
```
Run all cells, top to bottom. You can also open the notebook in VS Code and pick this folder's
`.venv` as the kernel. Training writes `model\tea_fertilizer_model.pkl`, the exact file the
chatbot reads. It also logs the run to MLflow (A3) and registers a new model version there.

**Test:**
```
.venv\Scripts\python -m pytest tests\
```
The tests check the model version marked `champion` in MLflow: its quality, its input format,
sensible behaviour (more land never means less fertilizer), and speed.

### A2. Tea Disease Classification

Folder: `ML Models\Tea Disease Classification`

**First time only:**
```
cd "ML Models\Tea Disease Classification"
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt jupyter matplotlib seaborn pandas scikit-learn pillow
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python dataset\download_dataset.py
```
- The second `pip` line installs the CPU version of PyTorch, the deep-learning library. It's a
  large download, roughly 200 MB.
- The last line downloads the leaf photos from Kaggle into `dataset\data\`. If it asks you to
  log in, create a token on your Kaggle account page (Settings, then "Create New Token") and
  run it again.

**Train:**
```
.venv\Scripts\python -m jupyter lab train_tea_disease_classifier.ipynb
```
This trains a ResNet18 model (a well-known image network, re-trained here on tea leaves) and
writes `model\tea_disease_resnet18.pkl`, the file the chatbot reads. A second notebook,
`train_tea_disease_classifier_vgg16.ipynb`, trains a larger VGG16 "challenger" model for
comparison. ResNet18 stays the one in use, because VGG16 is about 5 times slower per photo.

**Test:**
```
.venv\Scripts\python -m pytest tests\
```

### A3. MLflow UI (training runs and model versions)

MLflow is a tool that records every training run: its settings, its scores, and the model it
produced. Both ML projects write to one shared store, `ML Models\mlflow\mlflow.db`, so one web
page shows both.

Run it with the fertilizer project's environment (A1 must be set up first):
```
cd "ML Models\Tea Fertilizer Dosage"
.venv\Scripts\python -m mlflow ui --backend-store-uri sqlite:///../mlflow/mlflow.db --port 5000 --workers 1
```
Open http://127.0.0.1:5000. The two experiments, `tea-disease-classification` and
`tea-fertilizer-dosage`, are listed in the sidebar. Press Ctrl+C to stop it.

Two Windows pitfalls this command already avoids:
- **Three slashes.** The store must be written `sqlite:///...`. A `file://` address or four
  slashes fails with a "malformed path" error.
- **One worker.** Without `--workers 1`, MLflow crashes on Windows with `WinError 10022`.

**Promoting a model.** The tests trust the version with the `champion` alias. Right now that
is `tea-disease-resnet18` v2 and `tea-fertilizer-dosage-rf` v6. To promote a new version, open
it in the MLflow UI and add the alias `champion`. Or, from the same folder:
```
.venv\Scripts\python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///../mlflow/mlflow.db'); mlflow.MlflowClient().set_registered_model_alias('tea-fertilizer-dosage-rf', 'champion', '7')"
```
Change the model name and the version number (`'7'`) to the ones you want. The chatbot itself
reads the `.pkl` files in `model\`, so copy the new model there too (re-running the training
notebook does this).

---

## Part B - The chatbot (six variants)

### B0. Shared setup (do this once for all six)

Folder: `Chat_Bot`

```
cd Chat_Bot
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
```
Open `Chat_Bot\.env` in an editor and set at least `GEMINI_API_KEY=...`. The other keys are
optional (see section 2). Never commit `.env`: it holds your secret keys. It is already listed in
`.gitignore`.

**Build the knowledge bases.** RAG (retrieval-augmented generation) means the agent looks
answers up in a small database before replying. Build that database once per variant you plan
to run, and again whenever `db\data\knowledge\*.json` changes:
```
.venv\Scripts\python -m agent.db.vector.chroma.ingest_chroma --reset
```
This builds it for B1. For the other variants, use the line below and change `crewai` to that
variant's key (`langchain`, `a2a`, `crewai`, `hybrid` or `voice`):
```
.venv\Scripts\python -c "from agent_loader import mount_agent_variant; mount_agent_variant('crewai'); import sys, runpy; sys.argv=['ingest', '--reset']; runpy.run_module('agent.db.vector.chroma.ingest_chroma', run_name='__main__')"
```
The first run downloads a small embedding model (about 80 MB).

**The launcher.** Every variant starts the same way, from the `Chat_Bot` folder:
```
.venv\Scripts\python launch.py up <key>       # backend + UI (+ expert servers if it has them)
.venv\Scripts\python launch.py list           # every variant's ports and LangSmith project
```
Press **Ctrl+C** to stop everything that `up` started. The page can take 10-20 seconds to come
up the first time.

### B1. `agent`: LangGraph router

```
.venv\Scripts\python launch.py up agent
```
Open http://localhost:8510. The backend is on port 8001.

How it works: a first LLM step classifies the question (disease, fertilizer or company help),
then a fixed graph sends it to exactly one expert. It is the simplest and cheapest variant, but
a question that mixes two topics only gets one of them answered.

### B2. `langchain`: LangChain supervisor

```
.venv\Scripts\python launch.py up langchain
```
Open http://localhost:8511. The backend is on port 8002.

How it works: a supervisor LLM sees each expert as a tool. It can call several experts for a
mixed question and combine their answers.

### B3. `a2a`: agents over the A2A protocol

```
.venv\Scripts\python launch.py up a2a
```
Open http://localhost:8512. The backend is on port 8003. The launcher also starts the three
expert servers on ports 9101-9103 and waits a few seconds for them before starting the backend.

How it works: each expert is its own small web server. It describes itself in an **Agent
Card**, a JSON file you can open at http://localhost:9101/.well-known/agent-card.json. The
supervisor reads the cards and messages the experts over HTTP. To start only the expert
servers:
```
.venv\Scripts\python launch.py a2a-servers
```

### B4. `crewai`: CrewAI crew

```
.venv\Scripts\python launch.py up crewai
```
Open http://localhost:8513. The backend is on port 8004.

How it works: a CrewAI "manager" agent delegates the question to crew members (the experts),
each with a role, goal and backstory. CrewAI does not report to LangSmith by itself, so the
backend bridges its events there (`observability\langsmith_crewai.py`).

### B5. `hybrid`: multi-framework A2A, production-ready

```
.venv\Scripts\python launch.py up hybrid
```
Open http://localhost:8514. The backend is on port 8005. The launcher also starts three expert
servers, one process each:

| Expert | Framework | Port | Agent Card |
|---|---|---|---|
| Disease (also reads leaf photos) | CrewAI | 9201 | http://localhost:9201/.well-known/agent-card.json |
| Fertilizer | LangChain | 9202 | http://localhost:9202/.well-known/agent-card.json |
| Company help | LangGraph | 9203 | http://localhost:9203/.well-known/agent-card.json |

These ports differ from B3's, so B3 and B5 can run at the same time.

On top of the chat, this variant adds the production ("AIOps") tools. Run these from `Chat_Bot`:
```
.venv\Scripts\python "agent (crew AI , langchain)\run_hybrid.py" experts            # only the 3 expert servers
.venv\Scripts\python "agent (crew AI , langchain)\run_hybrid.py" smoke --ask        # health check: are all experts up, ready, answering?
.venv\Scripts\python "agent (crew AI , langchain)\run_hybrid.py" eval               # quality check: 19 test questions, needs the experts running
.venv\Scripts\python -m pytest "agent (crew AI , langchain)\tests" -q               # 39 automatic tests, no API key needed
```
- `smoke` is the check to run after every start or deploy. Without `--ask` it makes no LLM
  calls.
- `eval` takes about 10 minutes on the free Gemini tier. It fails if fewer than 80% of
  questions reach the right expert, and saves a report in `Chat_Bot\data\evals\`.

Useful pages while it runs:
- `http://localhost:9201/readyz` (or 9202, 9203) shows whether that expert can work right now
  (API key, model file, database).
- `http://localhost:8005/metrics` shows the live counters: calls per expert, circuit breakers
  (which stop calling a failing expert for a while), and blocked prompt-injection attempts.

More detail is in `Chat_Bot\agent (crew AI , langchain)\README.md`.

### B6. `voice`: Sinhala, Tamil or English, typed or spoken

```
.venv\Scripts\python launch.py up voice
```
Open http://localhost:8515. The backend is on port 8006.

How it works: the experts only understand English, so this variant puts a translation step on
each side of the B1 router. It first works out the language (Sinhala and Tamil have their own
alphabets, so this needs no AI call) and translates the question into English. The experts
answer in English, and a last step translates the answer back. English questions skip both
steps, so they cost nothing extra; Sinhala/Tamil questions need two extra LLM calls, so they are
slower on the free tier.

Voice: click the microphone next to the paperclip, speak, then click stop - the recording is
sent at once. The backend turns speech into text (**STT**, speech-to-text) before running the
graph, and with **🔊 Read answers aloud** on, turns the answer into an MP3 (**TTS**,
text-to-speech) that plays under the reply. In the sidebar, pick the language (use this if you
type Sinhala or Tamil with Latin letters - the detector would read that as English) and toggle
the spoken answers.

Settings in `.env` (defaults need no extra keys): `STT_PROVIDER=gemini` (listen with the same
Gemini model) or `groq` (Groq's hosted Whisper, needs `GROQ_API_KEY`); `TTS_PROVIDER=gtts`
(Google Translate voices) or `edge` (Microsoft neural voices, including Sri Lankan ones).
Tests that need no API key: `.venv\Scripts\python -m pytest tests -q` from `Chat_Bot`.

### B7. Running pieces separately, or everything at once

```
.venv\Scripts\python launch.py backend <key>    # only the backend
.venv\Scripts\python launch.py ui <key>         # only the UI (its backend must be running)
.venv\Scripts\python launch.py up all           # all six variants at once
```
`up all` makes many LLM calls in parallel if you chat in several UIs. On the free Gemini tier,
expect slower answers or quota errors.

---

## 3. Extras

### 3.1 Seeing what the agents did (LangSmith)

In `Chat_Bot\.env` set:
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<your key>
```
Restart the variant. Each answer in the UI then gets a link to its trace: every LLM call and
tool call, step by step. Each variant uses its own LangSmith project (`launch.py list` shows the
names). To rename one, set `LANGSMITH_PROJECT_<KEY>`, for example `LANGSMITH_PROJECT_HYBRID=...`.

### 3.2 The built-in Observability page

Every UI has an **Observability** page in its left menu. It shows latency, which experts were
used, and LLM and tool usage, from a local database in `Chat_Bot\data\`. No extra setup is needed.

### 3.3 Asking a backend directly (without the UI)

```
curl.exe -X POST http://127.0.0.1:8005/chat -F "message=How much fertilizer should I use?"
curl.exe -X POST http://127.0.0.1:8005/chat -F "message=What is this?" -F "image=@leaf.jpg"
```
Change the port to the variant you are running (8001-8005). Type `curl.exe`, not `curl`:
in Windows PowerShell, plain `curl` is a different command that doesn't understand `-F`.

### 3.4 Which farmer am I?

The backend always answers as farmer **id 3** (Balangoda, 2.5 acres), set in
`Chat_Bot\backend\routes\chat.py`. Fertilizer answers ("when did I last fertilize?") come from
that farmer's records in `tea_farm.db`.

### 3.5 Running B5 in Docker (with Prometheus and Grafana)

You need Docker Desktop for this. From the repo root:
```
cd "Chat_Bot\agent (crew AI , langchain)\deploy"
copy .env.example .env
```
Open `deploy\.env` and set `GEMINI_API_KEY` and `A2A_AUTH_TOKEN`. The token is a password the
agents use with each other. Make one with:
```
python -c "import secrets; print(secrets.token_urlsafe(32))"
```
Then start everything:
```
docker compose up -d --build
```
| Service | Address |
|---|---|
| Chat UI | http://localhost:8514 |
| Backend | http://localhost:8005 |
| Prometheus (metrics and alerts) | http://localhost:9090 |
| Alertmanager | http://localhost:9093 |
| Grafana (dashboard "Tea agents (A2A)", login admin / admin) | http://localhost:3000 |

To stop it, run `docker compose down`.

---

## 4. Troubleshooting

| What you see | Why | Fix |
|---|---|---|
| `Only one usage of each socket address` / port already in use | An earlier run is still alive | Close that terminal, or find it with `netstat -ano \| findstr :8005` and run `taskkill /PID <pid> /F` |
| Backend page says "offline" in the UI | The UI started before its backend, or the backend crashed | Look at the launcher's terminal for a red error. Start with `launch.py up <key>`, not `ui` alone |
| `GEMINI_API_KEY is not configured` | `.env` is missing or is in the wrong folder | It must be `Chat_Bot\.env`, and commands must run from `Chat_Bot` |
| Answers say "temporarily unavailable", or `429` / `RESOURCE_EXHAUSTED` in logs | Free-tier Gemini limit reached (requests per minute or per day) | Wait a minute and ask again. Add `GROQ_API_KEY` to get a backup model |
| `Chroma collection ... is empty` | The knowledge base was never built for this variant | Run the ingest command from B0 for that variant |
| `` `agent` was imported before mount_agent_variant() `` | A backend was started with plain `uvicorn` instead of the launcher | Use `launch.py backend <key>` |
| `'charmap' codec can't encode` | The Windows console can't print emoji from CrewAI | Use the launcher (it sets UTF-8). Or first run `set PYTHONIOENCODING=utf-8` (Command Prompt) or `$env:PYTHONIOENCODING="utf-8"` (PowerShell) |
| B3 or B5: an expert shows offline in the sidebar | Its server isn't running | B3: `launch.py a2a-servers`. B5: `run_hybrid.py experts`. Then click "Re-check" |
| MLflow: `WinError 10022` | MLflow started with several workers | Add `--workers 1` (A3) |
| ML tests are "skipped: no champion alias" | No version has been promoted yet | Promote one (A3) |

---

## 5. Quick reference: every port

| Port | What |
|---|---|
| 5000 | MLflow UI |
| 8001-8005 | Backends: agent, langchain, a2a, crewai, hybrid |
| 8510-8514 | UIs: agent, langchain, a2a, crewai, hybrid |
| 9101-9103 | B3 expert servers (disease, fertilizer, help) |
| 9201-9203 | B5 expert servers (disease: CrewAI, fertilizer: LangChain, help: LangGraph) |
| 9090 / 9093 / 3000 | Prometheus / Alertmanager / Grafana (B5 in Docker only) |
