"""Argument schemas for tools whose input the LLM writes (shared by the framework adapters).

The schema tells the LLM each argument's name and type. Facts the LLM must NOT write (the
farmer's id, the photo path) are never in a schema: the tools read them from the turn's state.
"""
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    query: str = Field(description="What to search for, in plain words.")


class LocationInput(BaseModel):
    location: str = Field(description="Location name, e.g. 'Kandy'.")
