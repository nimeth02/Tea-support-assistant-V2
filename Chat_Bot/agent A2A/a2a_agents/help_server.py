"""A2A server for the tea company help desk (wraps the HelpAgent RAG answerer)."""
from a2a.types import AgentSkill
from fastapi import FastAPI

from agent.a2a_agents.common import ExpertExecutor, ExpertRequest, build_a2a_app, build_agent_card
from agent.agents.help.help_agent import HelpAgent
from agent.config import settings


def agent_card(url: str = settings.A2A_HELP_AGENT_URL):
    return build_agent_card(
        name="Tea Company Help Agent",
        description=(
            "The tea company and its services: green leaf collection days and times, leaf quality "
            "and deductions, payments, advances, loans, supplier registration, complaints, policies."
        ),
        url=url,
        skills=[
            AgentSkill(
                id="company_help",
                name="Tea company help desk",
                description="Answers questions about the company's services and policies from its help database.",
                tags=["tea", "company", "payments", "collection"],
                examples=["When does the company collect green leaves?", "Can I get an advance?"],
            )
        ],
    )


class HelpExecutor(ExpertExecutor):
    def __init__(self):
        self.agent = HelpAgent()

    async def answer(self, request: ExpertRequest) -> str:
        result = await self.agent.help({"query": request.question, "intent": "company_help", "history": []})
        return result["response"]


def create_app() -> FastAPI:
    return build_a2a_app(agent_card(), HelpExecutor())
