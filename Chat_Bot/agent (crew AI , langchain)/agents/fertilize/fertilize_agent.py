"""Fertilizer expert - built with LangChain.

LangChain's create_agent makes a tool-calling agent: the LLM gets the tools' names and
descriptions through the provider's native function calling, decides which to call, reads the
results, and writes the answer. The farmer's id is passed as `context=` (not inside the prompt),
and the tools read it from there.

The LLM fallback chain (Gemini models, then Groq) comes from ModelFallbackMiddleware
(see services/llm_service.py), so this file doesn't need its own retry loop.
"""
import logging
from typing import Optional

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.messages import HumanMessage, ToolMessage

from agent.agents.base import ExpertAnswer
from agent.agents.fertilize.tools import FERTILIZE_TOOLS
from agent.prompt_templates import fertilize_agent_prompt
from agent.services.llm_service import agent_model_with_fallbacks
from agent.state.agent_state import ChatContext

logger = logging.getLogger("agent.fertilize")

FALLBACK_TEXT = (
    "Sorry, I couldn't process your fertilizer question right now. Please try again, or ask "
    "something like 'How much fertilizer should I use?' or 'When did I last fertilize?'"
)


class FertilizeAgent:
    name = "fertilize"
    framework = "langchain"

    def __init__(self):
        model, fallback_middleware = agent_model_with_fallbacks()
        self.agent = create_agent(
            model,
            tools=FERTILIZE_TOOLS,
            system_prompt=fertilize_agent_prompt,
            context_schema=ChatContext,
            middleware=[
                # Raise an error after 5 LLM calls so a confused model can't loop forever.
                ModelCallLimitMiddleware(run_limit=5, exit_behavior="error"),
                *fallback_middleware,  # next model in line when one fails (e.g. quota used up)
            ],
            name="fertilize_agent",
        )

    async def answer(self, question: str, user_id: int, image_path: Optional[str] = None) -> ExpertAnswer:
        try:
            result = await self.agent.ainvoke(
                {"messages": [HumanMessage(question)]}, context=ChatContext(user_id=user_id)
            )
        except Exception as e:
            logger.warning("fertilize_agent_failed error=%s", e)
            return ExpertAnswer(text=FALLBACK_TEXT, degraded=True)

        messages = result["messages"]
        tools_used = list(dict.fromkeys(m.name for m in messages if isinstance(m, ToolMessage)))
        model = (messages[-1].response_metadata or {}).get("model_name")
        return ExpertAnswer(text=messages[-1].text or FALLBACK_TEXT, tools_used=tools_used, model=model)
