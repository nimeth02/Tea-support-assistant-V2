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

`train_tea_disease_classifier.ipynb` logs every run to a local MLflow
tracking store (`mlflow.db`, a local SQLite file, gitignored) and registers
the resulting model under `tea-disease-resnet18` in the MLflow Model
Registry. The notebook still writes `model/tea_disease_resnet18.pkl` as
before (that's what the chatbot's loader reads), but each run's
params/metrics/artifacts and a versioned copy of the model now live in
MLflow too.

To browse runs and registered versions:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

then open <http://127.0.0.1:5000>.
