from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from agent.config import settings
import httpx
import os

# Shared across all LLM instances so the free-tier requests-per-minute quota isn't exceeded
gemini_rate_limiter = InMemoryRateLimiter(
    requests_per_second=settings.GEMINI_RPM / 60,
    check_every_n_seconds=0.5,
    max_bucket_size=1,
)

class LLMService:
    def __init__(self):
        try:
            if settings.LLM_PROVIDER.lower() == "gemini":
                self.llm = self._init_gemini()
            else:
                self.llm = self._init_groq()
        except Exception as e:
            print(f"Failed to initialize LLM: {str(e)}")
            raise

    def _init_gemini(self):
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not configured")

        def make(model):
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=settings.GEMINI_API_KEY,
                timeout=30.0,
                max_retries=1,  # fail over quickly; fallback models handle quota errors
                rate_limiter=gemini_rate_limiter
            )

        llm = make(settings.GEMINI_MODEL)
        fallback_models = [m.strip() for m in settings.GEMINI_FALLBACK_MODELS.split(",") if m.strip()]
        fallbacks = [make(m) for m in fallback_models]

        # Last resort, a different provider entirely: if every Gemini model above times out or
        # errors (e.g. a Gemini-wide slowdown), fall over to Groq's free tier instead of failing
        # the whole request. Only added when a Groq key is actually configured.
        if settings.GROQ_API_KEY:
            fallbacks.append(self._build_groq())
            fallback_models.append(f"groq:{settings.GROQ_MODEL}")

        if fallbacks:
            llm = llm.with_fallbacks(fallbacks)
        print(f"Gemini LLM initialized successfully ({settings.GEMINI_MODEL}, fallbacks: {fallback_models})")
        return llm

    def _build_groq(self) -> ChatGroq:
        if not settings.GROQ_API_KEY.startswith('gsk_'):
            raise ValueError("Invalid Groq API key format")
        return ChatGroq(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY,
            timeout=30.0,
            max_retries=2,
        )

    def _init_groq(self):
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not configured")
        llm = self._build_groq()
        print("Groq LLM initialized successfully")
        return llm

    def _test_connection(self):
        """Test the connection with a simple request"""
        try:
            test_prompt = "Hello"
            response = self.llm.invoke(test_prompt)
            print(response)
            print("LLM connection test successful")
        except Exception as e:
            print(f"LLM connection test failed: {str(e)}")
            raise RuntimeError(f"LLM service unavailable: {str(e)}")

    def get_llm(self):
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        return self.llm
