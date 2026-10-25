from crewai import LLM, Agent

from agent.agents.help.tools import get_help_tools
from agent.db.vector import get_vector_db
from agent.prompt_templates import help_prompt
from agent.state.agent_state import TeaChatState


class HelpAgent:
    """Company help desk expert: answers from the company-help knowledge base only."""

    ROLE = "Tea company help desk"

    def __init__(self):
        self.vector_db = get_vector_db("help")

    def build(self, llm: LLM, chat_state: TeaChatState) -> Agent:
        """Build the CrewAI agent for one chat turn (its tools share that turn's state)."""
        return Agent(
            role=self.ROLE,
            goal=(
                "Answer farmers' questions about the tea company - leaf collection, payments, "
                "advances, loans, registration and policies - using only the help database."
            ),
            backstory=help_prompt,
            tools=get_help_tools(self.vector_db, chat_state),
            llm=llm,
            allow_delegation=False,
            max_iter=4,
            verbose=False,
        )
