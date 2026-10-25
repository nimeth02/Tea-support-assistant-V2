## Layout

```
dataset/
  download_dataset.py   Downloads the Kaggle dataset into dataset/data/ (gitignored)
```

## Setup

```
pip install -r requirements.txt
python dataset/download_dataset.py
```

Dataset source: [bmshahriaalam/tealeafbd-tea-leaf-disease-detection](https://www.kaggle.com/datasets/bmshahriaalam/tealeafbd-tea-leaf-disease-detection) on Kaggle.

## Experiment tracking & model versioning

`train_tea_disease_classifier.ipynb` logs every run to the **shared**
MLflow tracking store at `../mlflow/mlflow.db` (see `ML Models/mlflow/`) -
the same store the fertilizer dosage notebook uses, under a separate
`tea-disease-classification` experiment - and registers the resulting
model under `tea-disease-resnet18` in the MLflow Model Registry. The
notebook still writes `model/tea_disease_resnet18.pkl` as before (that's
what the chatbot's loader reads), but each run's params/metrics/artifacts
and a versioned copy of the model now live in MLflow too.

To browse runs and registered versions, see `ML Models/mlflow/README.md`.

## Testing

```
python -m pytest tests/
```

`tests/test_model.py` fetches whatever version carries the `champion`
alias in the Model Registry and tests *that* - not whatever file happens
to be sitting in `model/` on this machine. Four layers, plus one optional
test:

- **Quality gate** - the champion's own logged `test_accuracy` must clear
  an agreed floor.
- **Model contract** - the registered run's `class_names` still match the
  7 expected classes, and the model still outputs a valid probability
  distribution over them.
- **Behavioral** - determinism (same photo in, same prediction out, twice).
- **Latency** - inference must stay under 75ms/image (measured baseline:
  ~21-33ms; the VGG16 challenger measured 138.9ms, which this ceiling would
  have caught).
- **Real-photo regression** (skipped if the Kaggle dataset isn't
  downloaded) - a known healthy-leaf photo should still predict "Healthy
  leaf".

If no version has the `champion` alias yet, the suite skips with a clear
reason instead of failing - promote a version first (see
`ML Models/mlflow/README.md`).
