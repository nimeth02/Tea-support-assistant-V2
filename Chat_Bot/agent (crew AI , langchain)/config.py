"""All settings for the multi-framework A2A agent system, read from Chat_Bot/.env.

Every value has a safe default, so a fresh checkout runs locally with only GEMINI_API_KEY set.
In production (Docker / Kubernetes) the same names are passed as environment variables.
"""
import os
from pathlib import Path
from typing import Dict, List

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root (the folder that holds Chat_Bot/ and "ML Models/"), found from this file's location:
# Chat_Bot/agent/config.py -> parents[2]. Used for the default model paths below.
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- Environment -----------------------------------------------------------------------
    # "dev" relaxes a few production checks (for example, running without an A2A auth token).
    ENVIRONMENT: str = "dev"
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # "json" (one JSON object per line, for log tools) or "text"

    # ---- LangSmith tracing (optional, LangChain/LangGraph parts only) -----------------------
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "tea-support-assistant"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    # ---- LLM ---------------------------------------------------------------------------------
    LLM_PROVIDER: str = "gemini"  # "gemini" or "groq"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"
    # Comma-separated models tried in order when the primary fails (e.g. daily quota exhausted)
    GEMINI_FALLBACK_MODELS: str = "gemini-3.1-flash-lite,gemini-3.5-flash"
    GEMINI_RPM: int = 5  # free-tier requests per minute (per process)
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    # ---- Vector DB (RAG) ---------------------------------------------------------------------
    VECTOR_DB_PROVIDER: str = "chroma"  # "chroma" (local) or "pinecone" (cloud)
    CHROMA_DISEASE_COLLECTION: str = "tea-disease"
    CHROMA_HELP_COLLECTION: str = "company-help"
    PINECONE_API_KEY: str = ""
    PINECONE_DISEASE_INDEX_NAME: str = "tea-disease"
    PINECONE_HELP_INDEX_NAME: str = "company-help"
    COHERE_API_KEY: str = ""

    # ---- Weather (weatherapi.com) ------------------------------------------------------------
    WHETHER_BASE_URL: str = "http://api.weatherapi.com/v1"
    WHETHER_API: str = ""

    # ---- ML model files ----------------------------------------------------------------------
    # Override these in Docker/Kubernetes, where the models are mounted somewhere else.
    FERTILIZER_MODEL_PATH: str = str(
        REPO_ROOT / "ML Models" / "Tea Fertilizer Dosage" / "model" / "tea_fertilizer_model.pkl"
    )
    DISEASE_MODEL_PATH: str = str(
        REPO_ROOT / "ML Models" / "Tea Disease Classification" / "model" / "tea_disease_resnet18.pkl"
    )

    # ---- A2A (Agent2Agent protocol) ----------------------------------------------------------
    # Base URL of each expert's A2A server. Its Agent Card is at <url>/.well-known/agent-card.json.
    A2A_DISEASE_AGENT_URL: str = "http://localhost:9201"    # CrewAI
    A2A_FERTILIZE_AGENT_URL: str = "http://localhost:9202"  # LangChain
    A2A_HELP_AGENT_URL: str = "http://localhost:9203"       # LangGraph
    # Comma-separated list the orchestrator talks to (default: the three above).
    A2A_AGENT_URLS: str = ""
    # Host the expert servers bind to. 127.0.0.1 locally; 0.0.0.0 inside a container.
    A2A_BIND_HOST: str = ""
    # Shared secret: the orchestrator sends it as "Authorization: Bearer <token>" and every
    # expert rejects A2A calls without it. Empty = no auth (only allowed when ENVIRONMENT=dev).
    A2A_AUTH_TOKEN: str = ""

    # ---- Resilience --------------------------------------------------------------------------
    A2A_TIMEOUT_SECONDS: float = 120.0      # one expert answer can take ~40s on the Gemini free tier
    A2A_CONNECT_RETRIES: int = 2            # extra tries when an expert can't be reached at all
    A2A_RETRY_BACKOFF_SECONDS: float = 0.5  # first wait between tries; doubles each time
    CIRCUIT_FAILURE_THRESHOLD: int = 3      # failures in a row before we stop calling an expert
    CIRCUIT_RESET_SECONDS: float = 30.0     # how long to wait before trying that expert again
    A2A_DISCOVERY_TTL_SECONDS: float = 30.0  # how often to look again for experts that were down
    EXPERT_TIMEOUT_SECONDS: float = 90.0    # max time one expert may spend on one question

    # ---- Guardrails --------------------------------------------------------------------------
    MAX_INPUT_CHARS: int = 2000
    MAX_OUTPUT_CHARS: int = 4000

    # ---- Evals -------------------------------------------------------------------------------
    EVAL_MIN_ROUTING_ACCURACY: float = 0.8  # the eval run fails below this (a CI quality gate)

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() in ("prod", "production", "staging")

    @property
    def a2a_agent_urls(self) -> List[str]:
        urls = self.A2A_AGENT_URLS or ",".join(self.expert_urls.values())
        return [u.strip().rstrip("/") for u in urls.split(",") if u.strip()]

    @property
    def expert_urls(self) -> Dict[str, str]:
        """Expert name -> its own A2A URL (used by the servers to know their own port)."""
        return {
            "disease": self.A2A_DISEASE_AGENT_URL,
            "fertilize": self.A2A_FERTILIZE_AGENT_URL,
            "help": self.A2A_HELP_AGENT_URL,
        }

    def gemini_models(self) -> List[str]:
        return [self.GEMINI_MODEL] + [
            m.strip() for m in self.GEMINI_FALLBACK_MODELS.split(",") if m.strip()
        ]


settings = Settings()

# LangChain reads these straight from os.environ (not from this Settings object), so mirror them
# in once at import time - before any LLM/agent call.
if settings.LANGSMITH_TRACING:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
