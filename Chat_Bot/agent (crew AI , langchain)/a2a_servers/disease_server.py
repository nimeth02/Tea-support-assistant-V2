"""A2A server for the tea disease / pest expert (CrewAI inside)."""
from a2a.types import AgentSkill
from fastapi import FastAPI

from agent.a2a_servers.common import ExpertExecutor, build_a2a_app, build_agent_card
from agent.config import settings
from agent.ops import health

NAME, FRAMEWORK = "disease", "crewai"
IMAGE_MODES = ["text/plain", "image/jpeg", "image/png"]


def agent_card(url: str = settings.A2A_DISEASE_AGENT_URL):
    return build_agent_card(
        name="Tea Disease Agent",
        description=(
            "Tea plant diseases and pests: spots or discoloured leaves, insects, fungus, wilting, "
            "dieback, damaged stems or roots, and how to treat or prevent them. Also identifies "
            "the disease in an attached tea leaf photo."
        ),
        url=url,
        framework=FRAMEWORK,
        skills=[AgentSkill(
            id="disease",
            name="Diagnose tea diseases and pests",
            description="Diagnoses from a description or a leaf photo, with treatment and prevention advice.",
            tags=["tea", "disease", "pest", "image", FRAMEWORK],
            examples=["My tea leaves have small pale blisters, what is it?"],
            input_modes=IMAGE_MODES,
        )],
        input_modes=IMAGE_MODES,  # accepts leaf photos
    )


def _make_expert():
    from agent.agents.disease.disease_agent import DiseaseAgent  # imports CrewAI: keep it lazy
    _install_crewai_langsmith_bridge()
    return DiseaseAgent()


def _install_crewai_langsmith_bridge() -> None:
    """CrewAI doesn't report to LangSmith by itself. When tracing is on, reuse the project's
    CrewAI -> LangSmith event bridge (Chat_Bot/observability/langsmith_crewai.py), so the crew's
    LLM and tool steps show up inside the orchestrator's trace. Skipped quietly if not available
    (for example in a slim container without the observability package)."""
    if not settings.LANGSMITH_TRACING:
        return
    try:
        from observability import langsmith_crewai
        langsmith_crewai.install()
    except Exception as e:  # tracing is optional; never block the expert on it
        import logging
        logging.getLogger("agent.disease").warning("crewai_langsmith_bridge_unavailable error=%s", e)


def create_app() -> FastAPI:
    return build_a2a_app(
        agent_card(),
        ExpertExecutor(_make_expert, NAME, FRAMEWORK),
        readiness_checks=[
            health.llm_configured,
            health.file_exists(settings.DISEASE_MODEL_PATH, "disease_model"),
            health.vector_db_ready("disease"),
        ],
    )
