from typing import List, Optional

from pydantic import BaseModel, Field


class TeaChatState(BaseModel):
    """State schema of the CrewAI flow that handles one chat turn.

    It is a Pydantic model, so the flow checks the inputs against it when a turn starts
    (e.g. a user_id that is not a number fails right away). CrewAI adds an `id` field itself.

    The tools get this same state object: they read `user_id` and `image_path` from it, so the
    LLM never has to pass (or can get wrong) the farmer's id or photo path, and they write the
    expert they belong to into `experts_used`, which becomes `intent` at the end.
    """

    # Inputs, set when the flow starts
    user_id: int = 0
    query: str = ""
    image_path: Optional[str] = None

    # Filled in while the crew works / when it finishes
    experts_used: List[str] = Field(default_factory=list)
    intent: str = ""
    response: str = ""
