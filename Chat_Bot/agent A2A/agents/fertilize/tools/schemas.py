"""Input validation for the fertilizer dosage model.

Industry pattern: validate real input at the service boundary, before it
reaches the model - the same job Pydantic does for every FastAPI request,
applied here to a LangChain tool instead of an HTTP endpoint. Without this,
a bad DB row (a null landSize, a negative plant count, a slope outside
0-10) would either crash deep inside sklearn with a cryptic error, or -
worse - silently produce a dosage number for nonsense inputs, since nothing
checks the array's contents are plausible before predicting.
"""
from pydantic import BaseModel, Field


class LandFeatures(BaseModel):
    """The model's 4 input features, validated before scaler.transform()."""

    land_size: float = Field(gt=0, le=1000, description="Land size in acres")
    plant_amount: int = Field(gt=0, le=10_000_000, description="Total tea plants on the land")
    degree: int = Field(ge=0, le=10, description="Slope, 0-10")
    days_since_last_fert: int = Field(ge=0, le=36500, description="Days since last fertilization")
