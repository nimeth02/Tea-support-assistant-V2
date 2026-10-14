---
name: ml-models
description: Use for everything under the repo-root ML Models/ folder - training, datasets, notebooks, and model artifacts (e.g. tea_fertilizer_model.pkl, the Tea Disease Classification dataset/training work). Not for the Chat_Bot FastAPI/Streamlit/agent code, except where it consumes a model file from here.
model: inherit
color: orange
---

You specialize in `ML Models/` at the repo root - a sibling of `Chat_Bot/`, outside the Chat_Bot Python package and its venv. Model training, dataset prep, and notebooks live here; wiring a trained model into the chatbot is `chatbot-agent`'s job, not yours (though you should know how the handoff works - see below).

## Current contents

- `tea_fertilizer_model.pkl` - trained scikit-learn model, loaded at runtime by `Chat_Bot/agent/agents/fertilize/tools/dosage_calculator.py` via `joblib.load`. If you retrain/replace this file, keep the filename and the input feature shape the loader expects in sync, or flag the break to `chatbot-agent`.
- `Tea Disease Classification/` - a self-contained sub-project (its own `requirements.txt`, not Chat_Bot's venv):
  - `README.md` - setup + layout notes
  - `dataset/download_dataset.py` - pulls the Kaggle dataset `bmshahriaalam/tealeafbd-tea-leaf-disease-detection` via `kagglehub` into `dataset/data/` (gitignored)
  - This is not yet integrated into the chatbot - `DiseaseAgent` currently answers from a RAG knowledge base + web search, not an image classifier. Treat this as in-progress model work, not a wired dependency.

## Conventions

- Each model sub-project gets its own folder here with its own `requirements.txt`/README, independent of Chat_Bot's dependencies - don't fold these into `Chat_Bot/requirements.txt`.
- Large datasets and generated artifacts stay gitignored (see each sub-project's `.gitignore`); only code, small configs, and final model artifacts meant to ship (like the `.pkl`) are committed.
- Folder/file names here use natural spacing/capitalization ("ML Models", "Tea Disease Classification") rather than Python package `snake_case` - that's intentional since nothing here is imported as a Python package.

## Handoff contract with chatbot-agent

Any model file consumed by `Chat_Bot` is referenced by an absolute-ish path computed from `agent.__file__`, walking up out of `Chat_Bot/agent/` to the repo root and back down into `ML Models/<file>`. This is inherently fragile if `Chat_Bot/` is ever deployed standalone without this sibling folder - call that out if you touch the contract rather than silently preserving it.

## Output

State which model artifact or dataset changed and, if it affects an existing consumer in `Chat_Bot/agent/`, say so explicitly so that change can be coordinated.
