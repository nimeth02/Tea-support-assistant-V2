from typing import Optional

from pydantic import BaseModel

class ChatResponse(BaseModel):
    response: str
    intent:str
    trace_id: Optional[str] = None
    latency_ms: Optional[int] = None
    variant: Optional[str] = None
    langsmith_url: Optional[str] = None  # link to this chat's trace, when LangSmith tracing is on

    # Multilingual + voice variant only (None for the other variants)
    language: Optional[str] = None          # language the answer is written in: "en", "si" or "ta"
    original_query: Optional[str] = None    # the question exactly as the user typed or spoke it
    translated_query: Optional[str] = None  # the English version the experts saw, when it was translated
    transcript: Optional[str] = None        # what speech recognition heard, for voice messages
    audio_url: Optional[str] = None         # GET this backend path for the spoken answer (MP3), when asked for
