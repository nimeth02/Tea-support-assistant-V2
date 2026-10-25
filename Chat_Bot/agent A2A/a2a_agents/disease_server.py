"""A2A server for the tea disease / pest expert (wraps the LangChain DiseaseAgent)."""
from a2a.types import AgentSkill
from fastapi import FastAPI

from agent.a2a_agents.common import ExpertExecutor, ExpertRequest, build_a2a_app, build_agent_card
from agent.agents.disease.disease_agent import DiseaseAgent
from agent.config import settings


def agent_card(url: str = settings.A2A_DISEASE_AGENT_URL):
    return build_agent_card(
        name="Tea Disease Agent",
        description=(
            "Tea plant diseases and pests: spots or discoloured leaves, insects, fungus, wilting, "
            "dieback, damaged stems or roots, and how to treat or prevent them. Also identifies "
            "the disease in an attached tea leaf photo."
        ),
        url=url,
        skills=[
            AgentSkill(
                id="disease",
                name="Diagnose tea diseases and pests",
                description="Diagnoses from a description or a leaf photo, with treatment and prevention advice.",
                tags=["tea", "disease", "pest", "image"],
                examples=["My tea leaves have small pale blisters, what is it?"],
                input_modes=["text/plain", "image/jpeg", "image/png"],
            )
        ],
        input_modes=["text/plain", "image/jpeg", "image/png"],  # accepts leaf photos
    )


class DiseaseExecutor(ExpertExecutor):
    def __init__(self):
        self.agent = DiseaseAgent()

    async def answer(self, request: ExpertRequest) -> str:
        result = await self.agent.disease({
            "userId": request.user_id,
            "query": request.question,
            "image_path": request.image_path,
            "intent": "disease",
            "history": [],
        })
        return result["response"]


def create_app() -> FastAPI:
    return build_a2a_app(agent_card(), DiseaseExecutor())
