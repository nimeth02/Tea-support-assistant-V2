import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LangSmith tracing (optional): auto-traces every LLM/tool/graph-node call LangChain makes.
    # Get a free API key at https://smith.langchain.com - leave LANGSMITH_TRACING off (default) to
    # skip it entirely; it runs alongside the existing Prometheus/SQLite observability, not instead of it.
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "tea-support-assistant"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    # LLM provider: "gemini" or "groq"
    LLM_PROVIDER: str = "gemini"

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"
    # Comma-separated models tried in order when the primary fails (e.g. daily quota exhausted)
    GEMINI_FALLBACK_MODELS: str = "gemini-3.1-flash-lite,gemini-3.5-flash"
    GEMINI_RPM: int = 5  # free-tier requests per minute

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    # Vector DB for RAG (disease and company-help knowledge bases):
    #   "chroma"   - local, on disk at agent/db/data/chroma, no keys (build with: python -m agent.db.vector.chroma.ingest_chroma)
    #   "pinecone" - cloud indexes + Cohere embeddings (needs the keys below)
    VECTOR_DB_PROVIDER: str = "chroma"
    CHROMA_DISEASE_COLLECTION: str = "tea-disease"
    CHROMA_HELP_COLLECTION: str = "company-help"

    # Pinecone + Cohere (only when VECTOR_DB_PROVIDER=pinecone)
    PINECONE_API_KEY: str = ""
    PINECONE_DISEASE_INDEX_NAME: str = "tea-disease"
    PINECONE_HELP_INDEX_NAME: str = "company-help"
    COHERE_API_KEY: str = ""

    # weatherapi.com
    WHETHER_BASE_URL: str = "http://api.weatherapi.com/v1"
    WHETHER_API: str = ""

    # --- Multilingual + voice (this variant only) -------------------------------------------
    # Speech-to-text (turning a recorded question into text):
    #   "gemini" - send the audio to the same Gemini chat model (no extra key; understands Sinhala,
    #              Tamil and English)
    #   "groq"   - Groq's hosted Whisper model (needs GROQ_API_KEY; fast and accurate)
    STT_PROVIDER: str = "gemini"
    GROQ_WHISPER_MODEL: str = "whisper-large-v3"
    # Text-to-speech (reading the answer out loud):
    #   "gtts" - Google Translate voices, no key needed (default)
    #   "edge" - Microsoft Edge neural voices, no key needed, more natural (incl. Sri Lankan voices)
    TTS_PROVIDER: str = "gtts"
    # Longest answer (in characters) we turn into speech; longer answers are cut with a short note.
    TTS_MAX_CHARS: int = 1500


settings = Settings()

# LangChain reads these straight from os.environ (not from this Settings object), so mirror them
# in here once at import time - before any LLM/agent call - rather than requiring the shell to export them.
if settings.LANGSMITH_TRACING:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
