## What this is

Trains `model/tea_fertilizer_model.pkl`, consumed by
`Chat_Bot/agent/agents/fertilize/tools/dosage_calculator.py`: a
`StandardScaler` -> `PolynomialFeatures(degree=2)` -> `RandomForestRegressor`
pipeline that predicts fertilizer dosage (kg) from land size, plant count,
slope, and days since last fertilization.

**There is no real historical dataset for this model.** `dataset/generate_dataset.py`
makes one up from a documented formula (see its docstring) - it's a
placeholder for the training *pipeline*, not a source of real agronomic
recommendations. Swap in real data the moment it exists; everything else
(pipeline shape, MLflow tracking, output path) stays the same.

## Layout

```
dataset/
  generate_dataset.py   Generates the synthetic dataset into dataset/data/ (gitignored)
```

Same two-folder (`dataset/` + `model/`) layout as the Tea Disease
Classification project.

## Setup

```
pip install -r requirements.txt
python dataset/generate_dataset.py
```

Then run `train_tea_fertilizer_model.ipynb` top to bottom. It writes the
trained bundle to `model/tea_fertilizer_model.pkl`, overwriting whatever
was there - that's the exact path the chatbot reads from.

## Experiment tracking & model versioning

Every run logs to the **shared** MLflow tracking store at
`../mlflow/mlflow.db` (see `ML Models/mlflow/`) - the same store the
disease classifier notebook uses, under a separate `tea-fertilizer-dosage`
experiment - and registers the trained model under
`tea-fertilizer-dosage-rf` in the MLflow Model Registry.

To browse runs and registered versions, see `ML Models/mlflow/README.md`.
