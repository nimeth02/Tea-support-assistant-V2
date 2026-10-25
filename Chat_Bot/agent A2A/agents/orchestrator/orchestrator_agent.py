from typing import Any, Dict, List, Optional

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware, after_agent
from langchain_core.messages import HumanMessage, ToolMessage

from agent.agents.orchestrator.a2a_tools import RemoteAgent, a2a_tool, discover_agents
from agent.config import settings
from agent.prompt_templates.orchestrator_prompt import orchestrator_system_prompt
from agent.services.llm_service import LLMService
from agent.state.agent_state import ChatContext, TeaAgentState


def make_record_intent(intents: Dict[str, str]) -> AgentMiddleware:
    """Middleware that, when the agent finishes, stores which remote agents it used as `intent`.

    `intents` maps tool name -> intent label (from each agent's card), e.g.
    "ask_tea_disease_agent" -> "disease". Result: "disease", "disease+fertilize", or "unknown".
    """

    @after_agent(state_schema=TeaAgentState)
    def record_intent(state: TeaAgentState, runtime) -> Dict[str, Any]:
        used = [
            intents[m.name]
            for m in state["messages"]
            if isinstance(m, ToolMessage) and m.name in intents
        ]
        return {"intent": "+".join(dict.fromkeys(used)) or "unknown"}

    return record_intent


class OrchestratorAgent:
    """Supervisor agent that orchestrates remote expert agents over the A2A protocol.

    On the first message it reads each configured agent's A2A Agent Card and turns every agent
    into a tool. The LLM then decides on its own which agents to message - one, several, or none
    for an off-topic question - and writes one reply from their answers.
    """

    def __init__(self, agent_urls: Optional[List[str]] = None,
                 http_client: Optional[httpx.AsyncClient] = None):
        self.agent_urls = agent_urls or settings.a2a_agent_urls
        self.http_client = http_client or httpx.AsyncClient(timeout=settings.A2A_TIMEOUT_SECONDS)
        self.llm = LLMService().get_llm()
        self.remote_agents: List[RemoteAgent] = []
        self.agent = None  # built on first use, once the agent cards are known

    async def _ensure_agent(self) -> None:
        """Discover the remote agents and build the supervisor (once per OrchestratorAgent)."""
        if self.agent is not None:
            return
        self.remote_agents = await discover_agents(self.agent_urls, self.http_client)
        if not self.remote_agents:
            raise RuntimeError(f"No A2A agents reachable at {self.agent_urls} - are they running?")

        self.agent = create_agent(
            self.llm,
            tools=[a2a_tool(a) for a in self.remote_agents],
            system_prompt=orchestrator_system_prompt,
            state_schema=TeaAgentState,
            context_schema=ChatContext,
            middleware=[
                # Raise an error after 6 LLM calls so a confused model can't loop forever.
                ModelCallLimitMiddleware(run_limit=6, exit_behavior="error"),
                make_record_intent({a.tool_name: a.intent for a in self.remote_agents}),
            ],
            name="orchestrator",
        )

    async def process_chat(self, message: str, userId: int, image_path: str = None,
                            callbacks=None) -> Dict[str, Any]:
        """Run the supervisor for a user message and return {"response": ..., "intent": ...}.

        image_path, when given, is the local path of a leaf photo the user uploaded. The
        supervisor is only told a photo is attached; the photo itself is sent as file bytes
        to the agents whose card accepts images.
        """
        try:
            print("Processing chat request")
            print(message)
            context = ChatContext(user_id=userId, image_path=image_path)
            await self._ensure_agent()

            text = message
            if context.image_path:
                text += "\n\n(The farmer attached a photo of a tea leaf.)"

            result = await self.agent.ainvoke(
                {"messages": [HumanMessage(text)]},
                context=context,
                config={"callbacks": callbacks or []},
            )

            return {
                "response": result["messages"][-1].text,
                "intent": result.get("intent", "unknown"),
            }
        except Exception as e:
            print(f"Error processing chat: {str(e)}")
            return {
                "response": f"Error processing your request: {message}",
                "intent": "error",
            }
