## Shared MLflow tracking store

One MLflow tracking store for every `ML Models/` sub-project, so you run a
single UI instead of one server per model. Each project writes to its own
**experiment** inside this same store (that's the normal way MLflow keeps
multiple projects apart) and registers its own models in the shared
**Model Registry**:

| Project | Experiment | Registered model |
|---|---|---|
| Tea Disease Classification | `tea-disease-classification` | `tea-disease-resnet18` |
| Tea Fertilizer Dosage | `tea-fertilizer-dosage` | `tea-fertilizer-dosage-rf` |

Nothing in this folder is committed to git except this README and
`.gitignore` - `mlflow.db` (tracking data) and `mlruns/` (artifacts) are
generated locally by running either project's notebook.

## Viewing it

```
cd "ML Models/mlflow"
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open <http://127.0.0.1:5000>. Switch between projects using the
**Experiments** list in the sidebar - no need to stop/restart the server.

## Adding another project here later

Point its notebook's `mlflow.set_tracking_uri(...)` at
`(Path("..") / "mlflow" / "mlflow.db").resolve()` (same pattern the two
existing notebooks use) and give it its own `mlflow.set_experiment(...)`
name. It'll show up in this same UI automatically.
