from typing import Any, Dict

from langchain.agents import AgentExecutor, create_react_agent

from agent.agents.disease.tools import get_disease_tools
from agent.db.vector import get_vector_db
from agent.prompt_templates import disease_agent_prompt
from agent.services.llm_service import LLMService


class DiseaseAgent:
    """ReAct agent that diagnoses tea diseases/pests from the Pinecone KB with a web-search fallback."""

    def __init__(self):
        self.vector_db = get_vector_db("disease")
        self.llm = LLMService().get_llm()
        self.tools = get_disease_tools(self.vector_db)
        self.agent = self._initialize_agent()
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def _initialize_agent(self):
        prompt = disease_agent_prompt.partial(
            tool_names=", ".join([tool.name for tool in self.tools])
        )
        return create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)

    async def disease(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query = state["query"]
            image_path = state.get("image_path")
            if image_path:
                query = f"{query}\n\n(Attached photo of the tea leaf - image file path: {image_path})"

            input_data = {
                "input": query,
                "userId": state.get("user_id", "unknown"),
            }

            result = await self.agent_executor.ainvoke(input_data)

            return {
                "intent": state["intent"],
                "history": state["history"],
                "query": state["query"],
                "response": result.get("output", "Sorry, I couldn’t determine the disease."),
            }
        except Exception as e:
            return {
                "intent": state["intent"],
                "history": state["history"],
                "query": state["query"],
                "response": f"Agent execution failed: {str(e)}",
            }
