from app.db.help_vector_db import PineconeDatabase
from langchain.tools import Tool
from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate
from typing import Dict, Any

class MaintanceService:
    def __init__(self):
        self.vector_db = PineconeDatabase()
        self.tools = self._initialize_tools()
        self.agent = self._initialize_agent()

    def _initialize_tools(self):
        """Define the two maintenance tools."""
        return [
            Tool(
                name="PruningGuide",
                func=self._pruning_guide,
                description="Provides pruning instructions based on plant age or season."
            ),
            Tool(
                name="IrrigationPlanner",
                func=self._irrigation_planner,
                description="Suggests irrigation schedules based on season or soil conditions."
            )
        ]

    def _pruning_guide(self, query: str) -> str:
        """Tool to provide pruning instructions."""
        response = "Prune tea plants lightly every 6 months during early spring."
        return response

    def _irrigation_planner(self, query: str) -> str:
        """Tool to suggest irrigation schedules."""
        schedule = "Irrigate weekly during the growing season."
        return schedule

    def _initialize_agent(self):
        """Initialize the LangChain ReAct agent."""
        prompt = PromptTemplate.from_template(
            """You are a maintenance expert for tea plants. Use the provided tools to answer the user's query.
            Query: {query}
            Chat History: {history}
            Answer concisely and accurately."""
        )
        # return create_react_agent(
        #     llm=self.llm,
        #     tools=self.tools,
        #     prompt=prompt
        # )    

    async def maintance(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the maintenance service query using the agent."""
        query = state["query"]
        
        # Run the agent
        agent_input = {
            "query": query,
            "history": state.get("history", [])
        }
        response = await self.agent.ainvoke(agent_input)
        
        # Format response
        return {
            "intent": state["intent"],
            "history": state["history"] + [f"User: {query}", f"Bot: {response['output']}"],
            "query": query,
            "response": response["output"]
        }