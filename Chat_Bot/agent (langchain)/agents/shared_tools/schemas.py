"""Input schemas for the single-input tools (passed to Tool(..., args_schema=...)).

With native tool calling the LLM fills in the tool's arguments as JSON, so each tool needs a
schema telling it the argument's name and type. A plain Tool(...) without one gives the model a
vague "args / kwargs" shape, and the model then sometimes sends a dict where the tool expects
text. One field per schema: Tool passes that field's value to its function as the only argument.
"""
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    query: str = Field(description="What to search for, in plain words.")


class UserIdInput(BaseModel):
    userId: int = Field(description="The farmer's userId (integer, e.g. 3).")


class ImagePathInput(BaseModel):
    image_path: str = Field(description="The local file path of the uploaded tea leaf photo.")


class LocationInput(BaseModel):
    location: str = Field(description="Location name, e.g. 'Kandy'.")
