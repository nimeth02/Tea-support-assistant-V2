"""Input schemas for tools whose input comes from the LLM (set as the tool's args_schema).

The schema tells the LLM the argument's name and type. Tools that only need the farmer's
user_id or photo path have no schema at all: they read those from the flow state instead.
"""
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    query: str = Field(description="What to search for, in plain words.")


class LocationInput(BaseModel):
    location: str = Field(description="Location name, e.g. 'Kandy'.")
