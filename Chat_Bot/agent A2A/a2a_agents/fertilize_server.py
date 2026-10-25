"""A2A server for the fertilizer expert (wraps the LangChain FertilizeAgent)."""
from a2a.types import AgentSkill
from fastapi import FastAPI

from agent.a2a_agents.common import ExpertExecutor, ExpertRequest, build_a2a_app, build_agent_card
from agent.agents.fertilize.fertilize_agent import FertilizeAgent
from agent.config import settings


def agent_card(url: str = settings.A2A_FERTILIZE_AGENT_URL):
    return build_agent_card(
        name="Tea Fertilizer Agent",
        description=(
            "Fertilizer type and dosage, when to fertilize, the farmer's own fertilization history, "
            "rain or weather forecast for the farmer's land, and details of the farmer's tea land "
            "(location, size, slope, number of plants). Needs the farmer's user_id in the metadata."
        ),
        url=url,
        skills=[
            AgentSkill(
                id="fertilize",
                name="Tea fertilizer advice",
                description="Dosage, timing and history of fertilizing, plus land data and weather for the farmer's land.",
                tags=["tea", "fertilizer", "weather", "land"],
                examples=["How much fertilizer should I use?", "When did I last fertilize?"],
            )
        ],
    )


class FertilizeExecutor(ExpertExecutor):
    def __init__(self):
        self.agent = FertilizeAgent()

    async def answer(self, request: ExpertRequest) -> str:
        result = await self.agent.fertilize({
            "userId": request.user_id,
            "query": request.question,
            "intent": "fertilize",
            "history": [],
        })
        return result["response"]


def create_app() -> FastAPI:
    return build_a2a_app(agent_card(), FertilizeExecutor())
