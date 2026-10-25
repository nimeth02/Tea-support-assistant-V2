"""LLM factory for both frameworks used in this project.

LangChain/LangGraph and CrewAI each need their own kind of model object, but they should use the
same models in the same fallback order, configured in one place (.env):

    agent_model_with_fallbacks()  primary model + ModelFallbackMiddleware, for create_agent
                     (the orchestrator and the fertilizer agent).
    langchain_llm()  a chat model with with_fallbacks(), for plain LLM calls (help agent).
    crewai_llms()    a list of CrewAI LLM objects, primary first. CrewAI has no fallback
                     chain, so the disease agent tries them one after another itself.

Fallback order: GEMINI_MODEL, then each of GEMINI_FALLBACK_MODELS, then Groq (only when a Groq
key is set), so a daily quota running out on one model doesn't take the bot down.
"""
import logging
from functools import lru_cache
from typing import List, Tuple

from agent.config import settings

logger = logging.getLogger("agent.llm")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"  # Groq serves an OpenAI-compatible API


@lru_cache(maxsize=1)
def _gemini_rate_limiter():
    """One shared limiter per process so the free-tier requests-per-minute quota is respected."""
    from langchain_core.rate_limiters import InMemoryRateLimiter

    return InMemoryRateLimiter(
        requests_per_second=settings.GEMINI_RPM / 60, check_every_n_seconds=0.5, max_bucket_size=1
    )


def _check_groq_key() -> None:
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured")
    if not settings.GROQ_API_KEY.startswith("gsk_"):
        raise ValueError("Invalid Groq API key format")


# ---- LangChain / LangGraph --------------------------------------------------------------------

def langchain_chat_models() -> List:
    """Plain LangChain chat models in the order to try them (primary first)."""
    from langchain_groq import ChatGroq

    def groq():
        _check_groq_key()
        return ChatGroq(model=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY,
                        timeout=30.0, max_retries=2)

    if settings.LLM_PROVIDER.lower() != "gemini":
        return [groq()]

    from langchain_google_genai import ChatGoogleGenerativeAI

    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured")

    def gemini(model: str):
        return ChatGoogleGenerativeAI(
            model=model, google_api_key=settings.GEMINI_API_KEY, timeout=30.0,
            max_retries=1,  # fail over quickly; the fallbacks handle quota errors
            rate_limiter=_gemini_rate_limiter(),
        )

    models = [gemini(m) for m in settings.gemini_models()]
    if settings.GROQ_API_KEY:
        models.append(groq())
    logger.info("langchain_models primary=%s fallbacks=%s", settings.GEMINI_MODEL, len(models) - 1)
    return models


def langchain_llm():
    """One runnable with the fallbacks built in, for plain (tool-less) LLM calls."""
    primary, *fallbacks = langchain_chat_models()
    return primary.with_fallbacks(fallbacks) if fallbacks else primary


def agent_model_with_fallbacks() -> Tuple[object, List]:
    """(primary model, middleware) for LangChain's create_agent.

    Agents get the fallbacks as ModelFallbackMiddleware, not as llm.with_fallbacks(): a
    with_fallbacks() model that also has tools bound loses its LLM steps in LangSmith traces
    ("No indexed run ID" warnings). The middleware gives the same fallback order cleanly.
    """
    from langchain.agents.middleware import ModelFallbackMiddleware

    primary, *fallbacks = langchain_chat_models()
    return primary, ([ModelFallbackMiddleware(*fallbacks)] if fallbacks else [])


# ---- CrewAI -----------------------------------------------------------------------------------

def crewai_llms() -> List:
    """CrewAI LLM objects in the order to try them."""
    from crewai import LLM

    def groq():
        _check_groq_key()
        # "openai/" picks CrewAI's OpenAI provider; base_url points it at Groq.
        return LLM(model=f"openai/{settings.GROQ_MODEL}", api_key=settings.GROQ_API_KEY,
                   base_url=GROQ_BASE_URL, timeout=30, max_retries=2)

    if settings.LLM_PROVIDER.lower() != "gemini":
        return [groq()]
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured")

    llms = [
        LLM(model=f"gemini/{m}", api_key=settings.GEMINI_API_KEY, timeout=30, max_retries=1)
        for m in settings.gemini_models()
    ]
    if settings.GROQ_API_KEY:
        llms.append(groq())
    return llms
