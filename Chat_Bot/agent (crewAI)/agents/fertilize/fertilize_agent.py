from crewai import LLM, Agent

from agent.agents.fertilize.tools import get_fertilize_tools
from agent.prompt_templates import fertilize_agent_prompt
from agent.state.agent_state import TeaChatState


class FertilizeAgent:
    """Fertilizer expert: dosage, weather, land data and fertilization history for the farmer."""

    ROLE = "Tea fertilizer expert"

    def build(self, llm: LLM, chat_state: TeaChatState) -> Agent:
        """Build the CrewAI agent for one chat turn (its tools read the farmer's user_id from the state)."""
        return Agent(
            role=self.ROLE,
            goal=(
                "Tell the farmer what, when and how much to fertilize, using their own land, "
                "fertilization history and weather data."
            ),
            backstory=fertilize_agent_prompt,
            tools=get_fertilize_tools(chat_state),
            llm=llm,
            allow_delegation=False,
            max_iter=5,  # stop after 5 reasoning steps so a confused model can't loop forever
            max_execution_time=60,  # seconds
            verbose=False,
        )
