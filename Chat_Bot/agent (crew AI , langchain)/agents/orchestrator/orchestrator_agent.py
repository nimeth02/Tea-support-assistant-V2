"""Orchestrator - a LangChain supervisor that talks to the experts only through A2A.

    farmer message -> guardrails -> supervisor LLM -> A2A calls -> guardrails -> reply
                                          |-- ask_tea_disease_agent    (CrewAI expert)
                                          |-- ask_tea_fertilizer_agent (LangChain expert)
                                          '-- ask_tea_company_help_agent (LangGraph expert)

The supervisor doesn't know or care which framework each expert uses: it only sees tools built
from their Agent Cards. It decides on its own which experts to ask (one, several, or none for an
off-topic message) and writes one reply from their answers.

Production notes:
  * The registry and the compiled supervisor are cached per process (module level), because
    the backend creates a new OrchestratorAgent for every chat request.
  * A missing expert doesn't fail the chat: its tool returns an "unavailable" note and the
    supervisor tells the farmer that part can't be answered now.
"""
import logging
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware, after_agent
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from agent.agents.orchestrator.a2a_client import AgentRegistry, RemoteAgent, a2a_tool
from agent.config import settings
from agent.ops import guardrails
from agent.ops.telemetry import ORCHESTRATOR_TURNS, set_trace_id
from agent.prompt_templates.orchestrator_prompt import orchestrator_system_prompt
from agent.services.llm_service import agent_model_with_fallbacks
from agent.state.agent_state import OrchestratorContext, TeaAgentState

logger = logging.getLogger("agent.orchestrator")

UNAVAILABLE_RESPONSE = (
    "Sorry, the assistant is not available right now. Please try again in a few minutes."
)
ERROR_RESPONSE = "Sorry, something went wrong while answering. Please try again."

_registry: Optional[AgentRegistry] = None
_supervisors: Dict[tuple, Any] = {}  # tool names -> compiled supervisor
_llm = None


def get_registry() -> AgentRegistry:
    global _registry
    if _registry is None:
        _registry = AgentRegistry(settings.a2a_agent_urls)
    return _registry


def intents_used(messages: List[BaseMessage], intents: Dict[str, str]) -> str:
    """Which experts were asked, e.g. "disease+fertilize"; "unknown" when none."""
    used = [intents[m.name] for m in messages if isinstance(m, ToolMessage) and m.name in intents]
    return "+".join(dict.fromkeys(used)) or "unknown"


def make_record_intent(intents: Dict[str, str]) -> AgentMiddleware:
    """When the supervisor finishes, store the experts it used as `intent` (for backend metrics)."""

    @after_agent(state_schema=TeaAgentState)
    def record_intent(state: TeaAgentState, runtime) -> Dict[str, Any]:
        return {"intent": intents_used(state["messages"], intents)}

    return record_intent


def expert_calls(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """The reply metadata of every expert call in this turn (framework, tools, latency, status)."""
    return [m.artifact for m in messages if isinstance(m, ToolMessage) and isinstance(m.artifact, dict)]


def _trace_id_from(callbacks) -> Optional[str]:
    """The backend passes its trace id inside its ObservabilityCallback; reuse it if present."""
    for cb in callbacks or []:
        if getattr(cb, "trace_id", None):
            return cb.trace_id
    return None


class OrchestratorAgent:
    def __init__(self, registry: Optional[AgentRegistry] = None):
        self.registry = registry or get_registry()

    def _supervisor(self, agents: List[RemoteAgent]):
        global _llm
        key = tuple(a.tool_name for a in agents)
        if key not in _supervisors:
            _llm = _llm or agent_model_with_fallbacks()
            model, fallback_middleware = _llm
            _supervisors[key] = create_agent(
                model,
                tools=[a2a_tool(self.registry, a) for a in agents],
                system_prompt=orchestrator_system_prompt,
                state_schema=TeaAgentState,
                context_schema=OrchestratorContext,
                middleware=[
                    # Raise an error after 6 LLM calls so a confused model can't loop forever.
                    ModelCallLimitMiddleware(run_limit=6, exit_behavior="error"),
                    *fallback_middleware,  # next model in line when one fails
                    make_record_intent({a.tool_name: a.intent for a in agents}),
                ],
                name="orchestrator",
            )
            logger.info("supervisor_built experts=%s", list(key))
        return _supervisors[key]

    async def process_chat(self, message: str, userId: int, image_path: str = None,
                           callbacks=None, trace_id: Optional[str] = None) -> Dict[str, Any]:
        """Answer one farmer message. Returns {"response", "intent", "experts", "trace_id"}.

        Same signature as the other agent variants, so the backend works unchanged.
        image_path is a leaf photo the farmer uploaded; it is sent as file bytes only to experts
        whose card accepts images. The supervisor itself is only told that a photo is attached.
        """
        trace_id = set_trace_id(trace_id or _trace_id_from(callbacks))

        check = guardrails.check_input(message)
        if not check.allowed:
            ORCHESTRATOR_TURNS.labels("blocked").inc()
            logger.info("input_blocked reason=%s", check.reason)
            return {"response": check.message, "intent": "blocked", "experts": [], "trace_id": trace_id}

        try:
            agents = await self.registry.agents()
            if not agents:
                ORCHESTRATOR_TURNS.labels("unavailable").inc()
                logger.error("no_experts_reachable urls=%s", self.registry.urls)
                return {"response": UNAVAILABLE_RESPONSE, "intent": "unavailable", "experts": [],
                        "trace_id": trace_id}

            text = check.text
            if image_path:
                text += "\n\n(The farmer attached a photo of a tea leaf.)"
            result = await self._supervisor(agents).ainvoke(
                {"messages": [HumanMessage(text)]},
                context=OrchestratorContext(user_id=userId, image_path=image_path, trace_id=trace_id),
                config={"callbacks": callbacks or [], "run_name": "orchestrator",
                        "metadata": {"trace_id": trace_id}},
            )
        except Exception:
            ORCHESTRATOR_TURNS.labels("error").inc()
            logger.exception("orchestrator_failed")
            return {"response": ERROR_RESPONSE, "intent": "error", "experts": [], "trace_id": trace_id}

        ORCHESTRATOR_TURNS.labels("ok").inc()
        return {
            "response": guardrails.clean_output(result["messages"][-1].text),
            "intent": result.get("intent", "unknown"),
            "experts": expert_calls(result["messages"]),
            "trace_id": trace_id,
        }
