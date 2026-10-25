"""A2A server for the tea company help desk (LangGraph inside)."""
from a2a.types import AgentSkill
from fastapi import FastAPI

from agent.a2a_servers.common import ExpertExecutor, build_a2a_app, build_agent_card
from agent.config import settings
from agent.ops import health

NAME, FRAMEWORK = "help", "langgraph"


def agent_card(url: str = settings.A2A_HELP_AGENT_URL):
    return build_agent_card(
        name="Tea Company Help Agent",
        description=(
            "The tea company and its services: green leaf collection days and times, leaf quality "
            "and deductions, payments, advances, loans, supplier registration, complaints, policies."
        ),
        url=url,
        framework=FRAMEWORK,
        skills=[AgentSkill(
            id="company_help",
            name="Tea company help desk",
            description="Answers questions about the company's services and policies from its help database.",
            tags=["tea", "company", "payments", "collection", FRAMEWORK],
            examples=["When does the company collect green leaves?", "Can I get an advance?"],
        )],
    )


def _make_expert():
    from agent.agents.help.help_agent import HelpAgent
    return HelpAgent()


def create_app() -> FastAPI:
    return build_a2a_app(
        agent_card(),
        ExpertExecutor(_make_expert, NAME, FRAMEWORK),
        readiness_checks=[health.llm_configured, health.vector_db_ready("help")],
    )
