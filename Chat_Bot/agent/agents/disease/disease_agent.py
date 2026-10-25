from typing import Any, Dict

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.messages import HumanMessage

from agent.agents.disease.tools import get_disease_tools
from agent.db.vector import get_vector_db
from agent.prompt_templates import disease_agent_prompt
from agent.services.llm_service import LLMService


class DiseaseAgent:
    """Tool-calling agent that diagnoses tea diseases/pests from the KB with a web-search fallback."""

    def __init__(self):
        self.vector_db = get_vector_db("disease")
        self.llm = LLMService().get_llm()
        self.tools = get_disease_tools(self.vector_db)
        self.agent = create_agent(
            self.llm,
            tools=self.tools,
            system_prompt=disease_agent_prompt,
            # Raise an error after 6 LLM calls so a confused model can't loop forever.
            middleware=[ModelCallLimitMiddleware(run_limit=6, exit_behavior="error")],
            name="disease_agent",
        )

    async def disease(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query = state["query"]
            image_path = state.get("image_path")
            if image_path:
                query = f"{query}\n\n(Attached photo of the tea leaf - image file path: {image_path})"

            result = await self.agent.ainvoke({"messages": [HumanMessage(query)]})

            return {
                "intent": state["intent"],
                "history": state["history"],
                "query": state["query"],
                "response": result["messages"][-1].text or "Sorry, I couldn’t determine the disease.",
            }
        except Exception as e:
            return {
                "intent": state["intent"],
                "history": state["history"],
                "query": state["query"],
                "response": f"Agent execution failed: {str(e)}",
            }
