import asyncio

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.messages import HumanMessage

from agent.agents.fertilize.tools import FERTILIZE_TOOLS
from agent.prompt_templates import fertilize_agent_prompt
from agent.services.llm_service import LLMService
from agent.state.agent_state import ChatContext

FALLBACK_RESPONSE = (
    "Sorry, I couldn’t process your request right now. "
    "Please try rephrasing your question or ask something like:\n"
    "- 'How much fertilizer should I use for my tea plants?'\n"
    "- 'What’s the weather forecast for Kandy?'\n"
    "- 'When was my last fertilization?'"
)


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

    async def answer(self, question: str, context: ChatContext) -> str:
        """Answer one fertilizer question; the farmer's userId goes in the message for the tools."""
        try:
            message = HumanMessage(f"{question}\n\nUserId: {context.user_id}")
            result = await asyncio.wait_for(self.agent.ainvoke({"messages": [message]}), timeout=60)

            print("agent result", result["messages"][-1])
            return result["messages"][-1].text or FALLBACK_RESPONSE
        except Exception:
            # User-friendly fallback response
            return FALLBACK_RESPONSE
