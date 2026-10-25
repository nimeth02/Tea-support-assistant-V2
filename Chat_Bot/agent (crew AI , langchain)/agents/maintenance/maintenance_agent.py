from typing import Any, Dict

from agent.agents.maintenance.tools import MAINTENANCE_TOOLS
from agent.db.vector import get_vector_db
from agent.prompt_templates import maintenance_prompt


class MaintenanceAgent:
    """Tea plant maintenance agent (pruning, irrigation). Not wired to an LLM yet."""

    def __init__(self):
        self.vector_db = get_vector_db("help")
        self.tools = MAINTENANCE_TOOLS
        self.agent = self._initialize_agent()

    def _initialize_agent(self):
        """Placeholder until an LLM is wired in - then build it with langchain.agents.create_agent."""
        prompt = maintenance_prompt
        # return create_agent(self.llm, tools=self.tools, system_prompt=prompt)

    async def maintance(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the maintenance query using the agent."""
        query = state["query"]

        agent_input = {
            "query": query,
            "history": state.get("history", []),
        }
        response = await self.agent.ainvoke(agent_input)

        return {
            "intent": state["intent"],
            "history": state["history"] + [f"User: {query}", f"Bot: {response['output']}"],
            "query": query,
            "response": response["output"],
        }
