import asyncio
from typing import Any, Dict

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.messages import HumanMessage

from agent.agents.fertilize.tools import FERTILIZE_TOOLS
from agent.prompt_templates import fertilize_agent_prompt
from agent.services.llm_service import LLMService


class FertilizeAgent:
    """Tool-calling agent for fertilizer dosage, weather, land data and fertilization history."""

    def __init__(self):
        self.llm = LLMService().get_llm()
        self.tools = FERTILIZE_TOOLS
        self.agent = create_agent(
            self.llm,
            tools=self.tools,
            system_prompt=fertilize_agent_prompt,
            # Raise an error after 5 LLM calls so a confused model can't loop forever.
            middleware=[ModelCallLimitMiddleware(run_limit=5, exit_behavior="error")],
            name="fertilize_agent",
        )

    async def fertilize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the fertilizer query using the agent."""
        query = state["query"]
        userId = state["userId"]

        try:
            # The tools take the userId as input, so it goes in the message for the model.
            message = HumanMessage(f"{query}\n\nUserId: {userId}")
            result = await asyncio.wait_for(self.agent.ainvoke({"messages": [message]}), timeout=60)
            output = result["messages"][-1].text

            print("agent result", output)

            return {
                "intent": state["intent"],
                "history": state["history"]
                + [f"User: {query}", f"Bot: {output}"],
                "query": query,
                "response": output,
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
