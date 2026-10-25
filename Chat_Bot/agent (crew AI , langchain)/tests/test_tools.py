"""Tool logic against the real read-only farm database and the real dosage model (no LLM)."""
from pathlib import Path

import pytest

from agent.agents.fertilize.tools.dosage_calculator import dosage_calculator_by_user
from agent.agents.fertilize.tools.fertilize_history import fertilize_history_by_user
from agent.agents.fertilize.tools.tea_land_data import tea_land_data_by_user
from agent.config import settings
from agent.ops import health


def test_land_data_for_known_and_unknown_farmer():
    [land] = tea_land_data_by_user(3)
    assert land["location"] == "Balangoda" and land["landSize"] == 2.5
    assert "error" in tea_land_data_by_user(9999)[0]


def test_history_is_a_readable_sentence():
    assert fertilize_history_by_user(3).startswith("You fertilized on")
    assert "No fertilization history" in fertilize_history_by_user(9999)


@pytest.mark.skipif(not Path(settings.FERTILIZER_MODEL_PATH).is_file(), reason="model file not present")
def test_dosage_prediction_uses_the_model():
    answer = dosage_calculator_by_user(3)
    assert "Recommended fertilizer dosage:" in answer and "kg" in answer
    assert "Could not find land data" in dosage_calculator_by_user(9999)


def test_readiness_checks_report_missing_files():
    ok, results = health.run_checks([health.file_exists("does/not/exist.pkl", "model"), health.tea_farm_db_readable])
    assert not ok
    assert results["model"]["ok"] is False
    assert results["tea_farm_db_readable"]["ok"] is True
