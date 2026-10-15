"""Tests for the tea leaf disease classifier, found through MLflow's Model
Registry's champion alias - not a hardcoded file path.

Same three layers as the fertilizer model's test suite:
1. Quality gate   - would this version even be allowed to ship?
2. Model contract - has the artifact's shape silently changed?
3. Behavioral     - does the model's behavior make basic sense?
Plus an optional real-photo regression test, skipped when the (gitignored,
not always downloaded) Kaggle dataset isn't present on this machine.
"""
from pathlib import Path

import mlflow
import mlflow.pytorch
import pytest
import torch
from PIL import Image
from torchvision import transforms

MODEL_NAME = "tea-disease-resnet18"
MLFLOW_DB = (Path(__file__).parent.parent / ".." / "mlflow" / "mlflow.db").resolve()
IMG_SIZE = 224
# Standard ImageNet normalization - a fixed convention the pretrained
# backbone expects, not an experiment-specific value, so it's not something
# logged per-run and is safe to hardcode here the same way the training
# notebook does.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EXPECTED_CLASSES = [
    "1. Tea algal leaf spot",
    "2. Brown Blight",
    "3. Gray Blight",
    "4. Helopeltis",
    "5. Red spider",
    "6. Green mirid bug",
    "7. Healthy leaf",
]
DATASET_DIR = Path(__file__).parent.parent / "dataset" / "data" / "teaLeafBD" / "teaLeafBD"

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")


@pytest.fixture(scope="session")
def client():
    return mlflow.client.MlflowClient()


@pytest.fixture(scope="session")
def champion_version(client):
    try:
        return client.get_model_version_by_alias(MODEL_NAME, "champion")
    except Exception:
        pytest.skip(f"No 'champion' alias set on {MODEL_NAME} - promote a version first")


@pytest.fixture(scope="session")
def champion_run(client, champion_version):
    return client.get_run(champion_version.run_id)


@pytest.fixture(scope="session")
def champion_model():
    model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}@champion")
    model.eval()
    return model


@pytest.fixture(scope="session")
def transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@pytest.fixture
def synthetic_image():
    """A plain in-memory image - enough to prove the pipeline runs and
    returns a sane shape, without depending on the real (gitignored,
    possibly not downloaded) dataset."""
    return Image.new("RGB", (300, 300), color=(80, 140, 70))


# ---- 1. Quality gate: does the champion version meet the agreed bar? ----

def test_champion_meets_quality_bar(champion_run):
    """Deliberately looser than what we've observed (~0.71-0.72) - this
    should catch a real regression in a future retrain, not every small
    fluctuation between runs."""
    metrics = champion_run.data.metrics
    assert metrics["test_accuracy"] > 0.60, (
        f"test_accuracy {metrics['test_accuracy']:.3f} is below the 0.60 floor"
    )


# ---- 2. Model contract: has the artifact's shape silently changed? ----

def test_champion_run_has_expected_classes(champion_run):
    logged = champion_run.data.params["class_names"]
    class_names = [c.strip() for c in logged.split(",")]
    assert class_names == EXPECTED_CLASSES


def test_predicts_a_probability_over_all_classes(champion_model, transform, synthetic_image):
    input_tensor = transform(synthetic_image).unsqueeze(0)
    with torch.no_grad():
        output = champion_model(input_tensor)
    assert output.shape == (1, len(EXPECTED_CLASSES))
    probs = torch.softmax(output, dim=1)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)


# ---- 3. Behavioral: does the model's behavior make basic sense? ----

def test_prediction_is_deterministic(champion_model, transform, synthetic_image):
    input_tensor = transform(synthetic_image).unsqueeze(0)
    with torch.no_grad():
        pred_a = champion_model(input_tensor).argmax(1).item()
        pred_b = champion_model(input_tensor).argmax(1).item()
    assert pred_a == pred_b


# ---- 4. Real-photo regression test - skipped if the dataset isn't present ----

@pytest.mark.skipif(not DATASET_DIR.exists(), reason="Kaggle dataset not downloaded")
def test_known_healthy_leaf_photo_predicts_healthy(champion_model, transform):
    healthy_dir = DATASET_DIR / "7. Healthy leaf"
    sample_path = next(healthy_dir.glob("*.*"))
    image = Image.open(sample_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = champion_model(input_tensor)
    predicted = EXPECTED_CLASSES[output.argmax(1).item()]
    assert predicted == "7. Healthy leaf"
