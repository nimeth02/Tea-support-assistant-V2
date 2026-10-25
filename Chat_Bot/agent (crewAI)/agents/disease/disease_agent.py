from crewai import LLM, Agent

from agent.agents.disease.tools import get_disease_tools
from agent.db.vector import get_vector_db
from agent.prompt_templates import disease_agent_prompt
from agent.state.agent_state import TeaChatState


class DiseaseAgent:
    """Tea disease / pest expert: diagnoses from the KB and the leaf-photo model, web search as fallback."""

    ROLE = "Tea disease and pest expert"

    def __init__(self):
        self.vector_db = get_vector_db("disease")

    def build(self, llm: LLM, chat_state: TeaChatState) -> Agent:
        """Build the CrewAI agent for one chat turn (its tools share that turn's state)."""
        return Agent(
            role=self.ROLE,
            goal=(
                "Diagnose tea plant diseases and pests and give treatment and prevention advice "
                "that the tool results support."
            ),
            backstory=disease_agent_prompt,
            tools=get_disease_tools(self.vector_db, chat_state),
            llm=llm,
            allow_delegation=False,
            max_iter=6,  # stop after 6 reasoning steps so a confused model can't loop forever
            verbose=False,
        )
