from typing import Optional

from pydantic import BaseModel

class ChatResponse(BaseModel):
    response: str
    intent:str
    trace_id: Optional[str] = None
    latency_ms: Optional[int] = None
    variant: Optional[str] = None
    langsmith_url: Optional[str] = None  # link to this chat's trace, when LangSmith tracing is on
