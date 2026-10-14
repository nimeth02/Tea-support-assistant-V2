from typing import Any, Dict

from langchain.agents import AgentExecutor, create_react_agent

from agent.agents.fertilize.tools import FERTILIZE_TOOLS
from agent.prompt_templates import fertilize_agent_prompt
from agent.services.llm_service import LLMService


class FertilizeAgent:
    """ReAct agent for fertilizer dosage, weather, land data and fertilization history."""

    def __init__(self):
        self.llm = LLMService().get_llm()
        self.tools = FERTILIZE_TOOLS
        self.agent = self._initialize_agent()

    def _initialize_agent(self):
        prompt = fertilize_agent_prompt.partial(
            tool_names=", ".join([tool.name for tool in self.tools])
        )
        return create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)

    async def fertilize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the fertilizer query using the agent."""
        query = state["query"]
        userId = state["userId"]

        try:
            executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,  # Enable verbose logging for debugging
                max_iterations=5,  # Limit iterations to prevent infinite loops
                timeout=60,
                handle_parsing_errors=True,
            )
            result = executor.invoke({"input": query, "userId": userId})

            print("agent result", result)

            return {
                "intent": state["intent"],
                "history": state["history"]
                + [f"User: {query}", f"Bot: {result['output']}"],
                "query": query,
                "response": result["output"],
            }

        except Exception as e:
            # User-friendly fallback response
            fallback_response = (
                "Sorry, I couldn’t process your request right now. "
                "Please try rephrasing your question or ask something like:\n"
                "- 'How much fertilizer should I use for my tea plants?'\n"
                "- 'What’s the weather forecast for Kandy?'\n"
                "- 'When was my last fertilization?'"
            )
            return {
                "intent": state["intent"],
                "history": state["history"] + [f"User: {query}", f"Bot: {fallback_response}"],
                "query": query,
                "response": fallback_response,
            }
