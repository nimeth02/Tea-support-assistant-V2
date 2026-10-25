from crewai import LLM, Agent

from agent.agents.maintenance.tools import get_maintenance_tools
from agent.prompt_templates import maintenance_prompt
from agent.state.agent_state import TeaChatState


class MaintenanceAgent:
    """Tea plant maintenance expert (pruning, irrigation).

    Not added to the crew yet: its tools still return fixed placeholder answers. To enable it,
    add `self.maintenance_agent.build(llm, state)` to the agents list in the orchestrator.
    """

    ROLE = "Tea plant maintenance expert"

    def build(self, llm: LLM, chat_state: TeaChatState) -> Agent:
        return Agent(
            role=self.ROLE,
            goal="Advise tea farmers on pruning and irrigation using the maintenance tools.",
            backstory=maintenance_prompt,
            tools=get_maintenance_tools(chat_state),
            llm=llm,
            allow_delegation=False,
            max_iter=4,
            verbose=False,
        )
