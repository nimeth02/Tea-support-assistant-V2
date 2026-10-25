from typing import List

from crewai import LLM

from agent.config import settings

GROQ_BASE_URL = "https://api.groq.com/openai/v1"  # Groq serves an OpenAI-compatible API


class LLMService:
    """Builds CrewAI LLM objects from settings: the primary model first, then the fallbacks.

    CrewAI has no built-in fallback chain (like LangChain's with_fallbacks), so get_llms()
    returns the models in the order to try; the orchestrator re-runs the crew with the next
    one when a run fails (e.g. the daily Gemini quota is used up).
    """

    def get_llms(self) -> List[LLM]:
        if settings.LLM_PROVIDER.lower() != "gemini":
            print("Groq LLM initialized successfully")
            return [self._groq()]

        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not configured")
        models = [settings.GEMINI_MODEL] + [
            m.strip() for m in settings.GEMINI_FALLBACK_MODELS.split(",") if m.strip()
        ]
        llms = [self._gemini(m) for m in models]

        # Last resort, a different provider entirely - only when a Groq key is configured.
        if settings.GROQ_API_KEY:
            llms.append(self._groq())
            models.append(f"groq:{settings.GROQ_MODEL}")
        print(f"Gemini LLM initialized successfully ({models[0]}, fallbacks: {models[1:]})")
        return llms

    def get_llm(self) -> LLM:
        return self.get_llms()[0]

    def _gemini(self, model: str) -> LLM:
        return LLM(
            model=f"gemini/{model}",
            api_key=settings.GEMINI_API_KEY,
            timeout=30,
            max_retries=1,  # fail over quickly; the fallback models handle quota errors
        )

    def _groq(self) -> LLM:
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not configured")
        if not settings.GROQ_API_KEY.startswith("gsk_"):
            raise ValueError("Invalid Groq API key format")
        # "openai/" picks CrewAI's OpenAI provider; base_url points it at Groq.
        return LLM(
            model=f"openai/{settings.GROQ_MODEL}",
            api_key=settings.GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
            timeout=30,
            max_retries=2,
        )
