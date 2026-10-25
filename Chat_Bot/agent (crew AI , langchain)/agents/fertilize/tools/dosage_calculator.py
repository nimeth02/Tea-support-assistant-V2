"""Fertilizer dosage (kg) for a farmer, predicted by the scikit-learn model in
ML Models/Tea Fertilizer Dosage/ (path set by FERTILIZER_MODEL_PATH)."""
import logging
from datetime import datetime
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from pydantic import ValidationError

from agent.agents.fertilize.tools.schemas import LandFeatures
from agent.config import settings
from agent.db.sqlite.tea_farm_db import tea_farm_db

logger = logging.getLogger("agent.tools.dosage")

DEFAULT_RESPONSE = (
    "Please make sure your tea land details are registered, or ask something like "
    "'How much fertilizer should I use?' again later."
)


@lru_cache(maxsize=1)
def load_model() -> dict:
    """Load the model bundle once per process (it used to be re-read from disk on every call)."""
    artifacts = joblib.load(settings.FERTILIZER_MODEL_PATH)
    logger.info("fertilizer_model_loaded path=%s", settings.FERTILIZER_MODEL_PATH)
    return artifacts


def _days_since_last_fertilization(user_id: int) -> int:
    history = tea_farm_db.get_fertilize_history_by_user(user_id)  # newest first
    if not history:
        return 365  # no history: treat it as a year ago
    latest = datetime.strptime(history[0]["fertilizedDate"], "%Y-%m-%d")
    return (datetime.now() - latest).days


def dosage_calculator_by_user(user_id: int) -> str:
    """Calculate the recommended fertilizer dosage for a farmer."""
    land_rows = tea_farm_db.get_tea_land_by_user(user_id)
    if not land_rows:
        return "Could not find land data. " + DEFAULT_RESPONSE
    land = land_rows[0]
    days_since = _days_since_last_fertilization(user_id)

    try:
        features = LandFeatures(
            land_size=land["landSize"], plant_amount=land["plantAmount"],
            degree=land["degree"], days_since_last_fert=days_since,
        )
    except ValidationError as e:
        logger.warning("invalid_land_data user_id=%s error=%s", user_id, e)
        return "Your registered land details look invalid or incomplete. " + DEFAULT_RESPONSE

    try:
        artifacts = load_model()
        model, scaler, poly = artifacts["model"], artifacts["scaler"], artifacts["poly"]
    except Exception as e:
        logger.error("fertilizer_model_load_failed error=%s", e)
        return "The dosage model is not available right now. " + DEFAULT_RESPONSE

    x = np.array([[features.land_size, features.plant_amount, features.degree, features.days_since_last_fert]])
    # The scaler was fitted on a DataFrame, so give it the same column names (avoids a warning
    # and guards against columns silently arriving in the wrong order).
    names = getattr(scaler, "feature_names_in_", None)
    if names is not None:
        x = pd.DataFrame(x, columns=names)
    dosage = round(float(model.predict(poly.transform(scaler.transform(x)))[0]), 2)
    logger.info("dosage_prediction user_id=%s dosage_kg=%s", user_id, dosage)

    return (
        f"Recommended fertilizer dosage: {dosage} kg\n"
        f"Calculation based on:\n"
        f"- Location: {land['location']}\n"
        f"- Land size: {land['landSize']} acres\n"
        f"- Plant count: {land['plantAmount']}\n"
        f"- Slope: {land['degree']}/10\n"
        f"- Days since last fertilization: {days_since}"
    )
