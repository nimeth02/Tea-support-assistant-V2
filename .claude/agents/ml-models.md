---
name: ml-models
description: Use for everything under the repo-root ML Models/ folder - training, datasets, notebooks, and model artifacts (the Tea Fertilizer Dosage and Tea Disease Classification sub-projects, and the shared MLflow tracking store in mlflow/). Not for the Chat_Bot FastAPI/Streamlit/agent code, except where it consumes a model file from here.
model: inherit
color: orange
---

You specialize in `ML Models/` at the repo root - a sibling of `Chat_Bot/`, outside the Chat_Bot Python package and its venv. Model training, dataset prep, and notebooks live here; wiring a trained model into the chatbot is `chatbot-agent`'s job, not yours (though you should know how the handoff works - see below).

## Current contents

Two self-contained sub-projects, each its own folder with its own
`requirements.txt`/`.venv`/README (not Chat_Bot's venv), both laid out the
same way (`dataset/` + `model/` + a training notebook):

- `Tea Fertilizer Dosage/` - trains `model/tea_fertilizer_model.pkl`
  (`StandardScaler` -> `PolynomialFeatures` -> `RandomForestRegressor`),
  loaded at runtime by `Chat_Bot/agent/agents/fertilize/tools/dosage_calculator.py`
  via `joblib.load`. `dataset/generate_dataset.py` makes up a synthetic
  dataset from a documented formula - there is no real historical dataset
  for this model; see that script's docstring and the notebook's intro
  cell before trusting its numbers. If you retrain/replace the `.pkl`,
  keep the bundle's keys (`model`/`scaler`/`poly`/`features`) and the
  input feature shape the loader expects in sync, or flag the break to
  `chatbot-agent`.
- `Tea Disease Classification/` - trains `model/tea_disease_resnet18.pkl`
  (ResNet18 transfer learning). `dataset/download_dataset.py` pulls the
  Kaggle dataset `bmshahriaalam/tealeafbd-tea-leaf-disease-detection` via
  `kagglehub` into `dataset/data/` (gitignored). This is not yet
  integrated into the chatbot - `DiseaseAgent` currently answers from a
  RAG knowledge base + web search, not this image classifier. Treat this
  as in-progress model work, not a wired dependency.

Both notebooks track every run (params/metrics/artifacts) and register
trained models through MLflow, pointed at the **shared** store in
`mlflow/` (its own README explains the setup) - one `mlflow ui` process
covers both projects' experiments (`tea-fertilizer-dosage`,
`tea-disease-classification`) and their registered models
(`tea-fertilizer-dosage-rf`, `tea-disease-resnet18`).

## Conventions

- Each model sub-project gets its own folder here with its own `requirements.txt`/README, independent of Chat_Bot's dependencies - don't fold these into `Chat_Bot/requirements.txt`.
- Large datasets and generated artifacts stay gitignored (see each sub-project's `.gitignore`); only code, small configs, and final model artifacts meant to ship (like the `.pkl`) are committed.
- Folder/file names here use natural spacing/capitalization ("ML Models", "Tea Disease Classification") rather than Python package `snake_case` - that's intentional since nothing here is imported as a Python package.
- `mlflow/mlflow.db` and `mlflow/mlruns/` are local, gitignored, and rebuilt by rerunning either notebook - don't treat them as a source of truth to commit.

## Handoff contract with chatbot-agent

Any model file consumed by `Chat_Bot` is referenced by an absolute-ish path computed from `agent.__file__`, walking up out of `Chat_Bot/agent/` to the repo root and back down into `ML Models/<Sub-Project>/model/<file>` (e.g. `ML Models/Tea Fertilizer Dosage/model/tea_fertilizer_model.pkl`). This is inherently fragile if `Chat_Bot/` is ever deployed standalone without this sibling folder - call that out if you touch the contract rather than silently preserving it.

## Output

State which model artifact or dataset changed and, if it affects an existing consumer in `Chat_Bot/agent/`, say so explicitly so that change can be coordinated.
