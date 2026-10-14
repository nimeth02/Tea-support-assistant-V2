"""DosageCalculator tool: predicts fertilizer dosage (kg) for a user with the scikit-learn model."""
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from langchain.tools import Tool

import agent
from agent.db.sqlite.tea_farm_db import tea_farm_db

MODEL_PATH = (
    Path(agent.__file__).resolve().parent.parent.parent
    / "ML Models" / "Tea Fertilizer Dosage" / "model" / "tea_fertilizer_model.pkl"
)

DEFAULT_RESPONSE = (
    "Please make sure your tea land details are registered, or ask something like "
    "'How much fertilizer should I use?' again later."
)


def _days_since_last_fertilization(user_id: int) -> int:
    history = tea_farm_db.get_fertilize_history_by_user(user_id)  # newest first
    if not history:
        return 365  # Default to 1 year if no history
    latest = datetime.strptime(history[0]["fertilizedDate"], "%Y-%m-%d")
    return (datetime.now() - latest).days


def dosage_calculator_by_user(userId) -> str:
    """Calculate the recommended fertilizer dosage for a userId."""
    print("dosage calculation")
    user_id = int(userId)
    print(user_id)

    land_rows = tea_farm_db.get_tea_land_by_user(user_id)
    land_data = land_rows[0] if land_rows else None
    if not land_data:
        return "Could not find land data. " + DEFAULT_RESPONSE

    days_since = _days_since_last_fertilization(user_id)

    try:
        model_artifacts = joblib.load(MODEL_PATH)
        model = model_artifacts["model"]
        scaler = model_artifacts["scaler"]
        poly = model_artifacts["poly"]
    except Exception as e:
        print(f"Model loading failed: {e}")
        return "Model error. " + DEFAULT_RESPONSE

    input_data = np.array(
        [[land_data["landSize"], land_data["plantAmount"], land_data["degree"], days_since]]
    )
    print(input_data)

    input_scaled = scaler.transform(input_data)
    input_poly = poly.transform(input_scaled)
    dosage = round(float(model.predict(input_poly)[0]), 2)
    print("dosage", dosage)

    return (
        f"Recommended fertilizer dosage: {dosage} kg\n"
        f"Calculation based on:\n"
        f"- Location: {land_data['location']}\n"
        f"- Land size: {land_data['landSize']} acres\n"
        f"- Plant count: {land_data['plantAmount']}\n"
        f"- Slope: {land_data['degree']}/10\n"
        f"- Days since last fertilization: {days_since}\n"
        f"(User ID: {user_id})"
    )


dosage_calculator_tool = Tool(
    name="DosageCalculator",
    func=dosage_calculator_by_user,
    description="Calculates fertilizer dosage in kg for a userId. Input: userId (integer, e.g., 3)",
)
