from typing import Optional

from pydantic import BaseModel

class ChatResponse(BaseModel):
    response: str
    intent:str
    trace_id: Optional[str] = None
    latency_ms: Optional[int] = None
