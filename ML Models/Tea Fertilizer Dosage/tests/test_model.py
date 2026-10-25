"""Tests for the fertilizer dosage model, found through MLflow's Model
Registry - not a hardcoded file path.

Industry pattern: ask "what's the champion version right now?" through the
registry, and test *that* - so these tests keep testing whatever is
actually promoted, through every future retrain, rather than whatever file
happens to be sitting in model/ on this machine.

Four layers, each catching a different kind of failure:
1. Quality gate   - would this version even be allowed to ship?
2. Model contract - has the artifact's shape silently changed?
3. Behavioral     - does the model's behavior make domain sense?
4. Latency        - did it quietly get much slower?
"""
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pytest

# ~11x the measured baseline (~4.3ms) - loose enough to not be flaky on a
# slower machine, tight enough to still fail if a future retrain (e.g. a
# heavier model swapped in) makes this noticeably slower without anyone
# noticing, the way VGG16 (138.9ms) silently would have for the disease
# classifier if nothing asserted a ceiling.
LATENCY_CEILING_MS = 50

MODEL_NAME = "tea-fertilizer-dosage-rf"
MLFLOW_DB = (Path(__file__).parent.parent / ".." / "mlflow" / "mlflow.db").resolve()

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
def champion_bundle(champion_version):
    """The full scaler+poly+model pipeline, downloaded from this run's own
    logged artifact - not read from the local model/ folder, so the test
    always checks the artifact tied to the champion version, wherever it's
    actually run (a CI machine wouldn't have model/ populated at all)."""
    local_path = mlflow.artifacts.download_artifacts(
        f"runs:/{champion_version.run_id}/bundle/tea_fertilizer_model.pkl"
    )
    return joblib.load(local_path)


def _predict(bundle, land_size, plant_amount, degree, days_since):
    x = np.array([[land_size, plant_amount, degree, days_since]])
    x_scaled = bundle["scaler"].transform(x)
    x_poly = bundle["poly"].transform(x_scaled)
    return round(float(bundle["model"].predict(x_poly)[0]), 2)


# ---- 1. Quality gate: does the champion version meet the agreed bar? ----

def test_champion_meets_quality_bar(champion_run):
    """Block promotion if the champion's own recorded metrics fall below an
    agreed floor. These floors are deliberately looser than what we've
    actually observed (mae~12.4, r2~0.98) - they should catch a real
    regression in a future retrain, not flag every tiny fluctuation."""
    metrics = champion_run.data.metrics
    assert metrics["r2"] > 0.90, f"r2 {metrics['r2']:.3f} is below the 0.90 floor"
    assert metrics["mae"] < 20, f"mae {metrics['mae']:.2f} kg is above the 20kg ceiling"
    assert metrics["rmse"] < 25, f"rmse {metrics['rmse']:.2f} kg is above the 25kg ceiling"


# ---- 2. Model contract: has the artifact's shape silently changed? ----

def test_bundle_has_expected_keys(champion_bundle):
    assert set(champion_bundle.keys()) == {"model", "scaler", "poly", "features"}


def test_bundle_features_match_app_expectation(champion_bundle):
    # dosage_calculator.py builds its input array in exactly this order -
    # if this list ever changes, that code breaks silently (wrong feature
    # in the wrong column), not loudly.
    assert champion_bundle["features"] == ["landSize", "plantAmount", "degree", "days_since_last_fert"]


def test_predicts_a_single_positive_dosage(champion_bundle):
    dosage = _predict(champion_bundle, land_size=2.5, plant_amount=25000, degree=6, days_since=120)
    assert isinstance(dosage, float)
    assert dosage > 0


# ---- 3. Behavioral: does the model's behavior make domain sense? ----

def test_prediction_is_deterministic(champion_bundle):
    a = _predict(champion_bundle, land_size=2.5, plant_amount=25000, degree=6, days_since=120)
    b = _predict(champion_bundle, land_size=2.5, plant_amount=25000, degree=6, days_since=120)
    assert a == b


def test_more_land_does_not_recommend_less_fertilizer(champion_bundle):
    """Directional expectation test: holding everything else fixed, a much
    bigger plot should never get recommended *less* fertilizer than a small
    one. A model that violated this would be making an obviously wrong
    recommendation even if its overall accuracy metric looked fine - the
    kind of mistake per-sample accuracy alone can't catch."""
    small = _predict(champion_bundle, land_size=1.0, plant_amount=10000, degree=5, days_since=120)
    large = _predict(champion_bundle, land_size=5.0, plant_amount=50000, degree=5, days_since=120)
    assert large > small


def test_longer_gap_since_fertilizing_does_not_recommend_less(champion_bundle):
    soon = _predict(champion_bundle, land_size=2.5, plant_amount=25000, degree=5, days_since=30)
    long_gap = _predict(champion_bundle, land_size=2.5, plant_amount=25000, degree=5, days_since=300)
    assert long_gap > soon


# ---- 4. Latency: did it quietly get much slower? ----

def test_inference_latency_under_threshold(champion_bundle):
    """We measured a 6.5x latency difference between two architectures once,
    by hand, in a comparison notebook (ResNet18 vs VGG16, for the other
    model). Nothing stopped that from happening again unnoticed - this
    makes "did it get much slower" an enforced pass/fail instead of
    something you'd only catch if you happened to time it yourself."""
    champion_bundle["model"]  # bundle already loaded; nothing to warm up beyond the first call
    _predict(champion_bundle, land_size=2.5, plant_amount=25000, degree=6, days_since=120)  # warm-up

    n = 20
    start = time.time()
    for _ in range(n):
        _predict(champion_bundle, land_size=2.5, plant_amount=25000, degree=6, days_since=120)
    elapsed_ms = (time.time() - start) / n * 1000

    assert elapsed_ms < LATENCY_CEILING_MS, (
        f"inference took {elapsed_ms:.1f} ms/prediction, over the {LATENCY_CEILING_MS}ms ceiling"
    )
