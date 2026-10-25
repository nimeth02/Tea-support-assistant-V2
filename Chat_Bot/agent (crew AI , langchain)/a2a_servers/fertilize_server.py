"""A2A server for the fertilizer expert (LangChain inside)."""
from a2a.types import AgentSkill
from fastapi import FastAPI

from agent.a2a_servers.common import ExpertExecutor, build_a2a_app, build_agent_card
from agent.config import settings
from agent.ops import health

NAME, FRAMEWORK = "fertilize", "langchain"


def agent_card(url: str = settings.A2A_FERTILIZE_AGENT_URL):
    return build_agent_card(
        name="Tea Fertilizer Agent",
        description=(
            "Fertilizer type and dosage, when to fertilize, the farmer's own fertilization history, "
            "rain or weather forecast for the farmer's land, and details of the farmer's tea land "
            "(location, size, slope, number of plants). Needs the farmer's user_id in the metadata."
        ),
        url=url,
        framework=FRAMEWORK,
        skills=[AgentSkill(
            id="fertilize",
            name="Tea fertilizer advice",
            description="Dosage, timing and history of fertilizing, plus land data and weather for the farmer's land.",
            tags=["tea", "fertilizer", "weather", "land", FRAMEWORK],
            examples=["How much fertilizer should I use?", "When did I last fertilize?"],
        )],
    )


def _make_expert():
    from agent.agents.fertilize.fertilize_agent import FertilizeAgent
    return FertilizeAgent()


def create_app() -> FastAPI:
    return build_a2a_app(
        agent_card(),
        ExpertExecutor(_make_expert, NAME, FRAMEWORK),
        readiness_checks=[
            health.llm_configured,
            health.file_exists(settings.FERTILIZER_MODEL_PATH, "fertilizer_model"),
            health.tea_farm_db_readable,
        ],
    )
