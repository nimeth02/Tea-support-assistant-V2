---
name: mlflow-ui
description: Starts the shared MLflow tracking UI for this repo's ML Models/ training projects (the tea-disease-classification and tea-fertilizer-dosage experiments). Use whenever the user asks to view MLflow runs, open the MLflow dashboard, check training results/metrics, or look at traces.
---

Start the shared MLflow UI server and report the URL back to the user.

## Why this needs a dedicated skill

Starting `mlflow ui` the naive way hits two real bugs on Windows - both were
hit (and fixed) while building this repo's MLflow setup, so they're worth
avoiding every time instead of re-debugging them:

1. **SQLite path format.** The backend store URI must look like
   `sqlite:///C:\full\windows\path\to\mlflow.db` - three slashes, then a
   plain Windows path with a drive letter. A `file://` URI or a four-slash
   variant (`sqlite:////c:/...`) fails with a malformed-path error.
2. **Multi-worker crash.** MLflow's server defaults to multiple worker
   processes. On Windows this crashes with `OSError: [WinError 10022]`,
   because the socket-sharing trick it relies on to split traffic across
   workers doesn't work the same way across spawned processes on Windows.
   Always pass `--workers 1`.

## Steps

1. Resolve the absolute path to `ML Models/mlflow/mlflow.db` from the repo
   root (this is the one shared tracking store both training projects use
   - see `ML Models/mlflow/README.md`).
2. Check whether anything is already listening on port 5000. If an old
   instance (or a duplicate) is still running, stop it first rather than
   launching a second one on the same port - two servers on one port causes
   flaky, inconsistent responses.
3. Start the server in the background:
   ```
   python -m mlflow ui --backend-store-uri "sqlite:///<absolute-path-to-mlflow.db>" --port 5000 --workers 1
   ```
   Use any Python environment that already has `mlflow` installed (e.g.
   one of the `ML Models/*/.venv` folders in this repo).
4. Wait a few seconds, then confirm it's actually responding
   (`curl http://127.0.0.1:5000/`) before telling the user it's ready -
   don't just assume the process starting means it bound the port
   successfully.
5. Report the URL back: **http://127.0.0.1:5000/**. Mention both
   experiments (`tea-disease-classification`, `tea-fertilizer-dosage`) live
   there, as separate entries in the sidebar - no need to run two servers.
