"""The orchestrator's side of A2A: find the experts, call them safely, wrap them as tools.

  AgentRegistry   reads every expert's Agent Card and keeps a client + circuit breaker for each.
                  Experts that were down are looked for again every A2A_DISCOVERY_TTL_SECONDS,
                  so an expert that starts late (or comes back after a crash) joins by itself.
  RemoteAgent     one expert: its card, its client and its circuit breaker.
  call()          sends one question with: the auth token, the farmer's id and trace id in the
                  metadata, the photo if the expert accepts images, a timeout, retries on
                  connection errors, and the circuit breaker. It never raises: when an expert
                  can't answer, it returns a short note the orchestrator LLM can pass on.
  a2a_tool()      wraps one expert as a LangChain tool whose name and description come from
                  its card, so a new expert needs no prompt or code change here.
"""
import asyncio
import logging
import mimetypes
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from a2a.client import A2ACardResolver, Client, ClientConfig, ClientFactory
from a2a.helpers import get_stream_response_text, new_raw_part, new_text_part
from a2a.types import AgentCard, Message, Role, SendMessageRequest
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from langchain.tools import BaseTool, ToolRuntime, tool
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, Field

from agent.config import settings
from agent.ops.resilience import CircuitBreaker, retry_on_connect_error
from agent.ops.telemetry import AGENTS_DISCOVERED, CLIENT_CALLS, CLIENT_LATENCY
from agent.state.agent_state import OrchestratorContext

logger = logging.getLogger("agent.a2a_client")

UNAVAILABLE_NOTE = (
    "[{name} is temporarily unavailable. Tell the farmer this part of their question can't be "
    "answered right now and to try again in a few minutes. Do not answer it yourself.]"
)


class AgentQuestion(BaseModel):
    question: str = Field(
        description="The farmer's question for this agent, rewritten so it makes sense on its own."
    )


@dataclass
class RemoteAgent:
    url: str
    card: AgentCard
    client: Client
    breaker: CircuitBreaker

    @property
    def intent(self) -> str:
        """By project convention, the id of the card's first skill."""
        return self.card.skills[0].id if self.card.skills else self.tool_name

    @property
    def tool_name(self) -> str:
        """e.g. "Tea Disease Agent" -> "ask_tea_disease_agent"."""
        return "ask_" + re.sub(r"[^a-z0-9]+", "_", self.card.name.lower()).strip("_")

    @property
    def accepts_images(self) -> bool:
        return any(mode.startswith("image/") for mode in self.card.default_input_modes)


class AgentRegistry:
    def __init__(self, urls: List[str], http_client: Optional[httpx.AsyncClient] = None):
        self.urls = urls
        headers = {"Authorization": f"Bearer {settings.A2A_AUTH_TOKEN}"} if settings.A2A_AUTH_TOKEN else {}
        self.http = http_client or httpx.AsyncClient(timeout=settings.A2A_TIMEOUT_SECONDS, headers=headers)
        self._factory = ClientFactory(ClientConfig(httpx_client=self.http, streaming=False))
        self._agents: Dict[str, RemoteAgent] = {}
        self._last_discovery = float("-inf")
        self._lock = asyncio.Lock()

    async def agents(self) -> List[RemoteAgent]:
        """The reachable experts, in URL order. Looks for missing ones again after the TTL."""
        missing = [u for u in self.urls if u not in self._agents]
        if missing and time.monotonic() - self._last_discovery >= settings.A2A_DISCOVERY_TTL_SECONDS:
            async with self._lock:
                await self._discover([u for u in self.urls if u not in self._agents])
        return [self._agents[u] for u in self.urls if u in self._agents]

    async def _discover(self, urls: List[str]) -> None:
        self._last_discovery = time.monotonic()
        for url in urls:
            try:
                card = await A2ACardResolver(self.http, url).get_agent_card()
            except Exception as e:
                logger.warning("a2a_agent_unreachable url=%s error=%s", url, e)
                continue
            name = card.skills[0].id if card.skills else card.name
            self._agents[url] = RemoteAgent(
                url=url, card=card, client=self._factory.create(card),
                breaker=CircuitBreaker(name, settings.CIRCUIT_FAILURE_THRESHOLD, settings.CIRCUIT_RESET_SECONDS),
            )
            logger.info("a2a_agent_found name=%s url=%s", card.name, url)
        AGENTS_DISCOVERED.set(len(self._agents))

    async def call(self, agent: RemoteAgent, question: str,
                   context: OrchestratorContext) -> Tuple[str, Dict[str, Any]]:
        """Ask one expert one question. Returns (answer text, reply metadata)."""
        label = agent.intent
        if not agent.breaker.allow():
            CLIENT_CALLS.labels(label, "circuit_open").inc()
            return UNAVAILABLE_NOTE.format(name=agent.card.name), {"agent": label, "status": "circuit_open"}

        message = self._build_message(agent, question, context)

        async def send() -> Tuple[str, Dict[str, Any]]:
            texts, meta = [], {}
            async for response in agent.client.send_message(SendMessageRequest(message=message)):
                texts.append(get_stream_response_text(response))
                if response.HasField("message"):
                    meta = MessageToDict(response.message.metadata)
            return "\n".join(t for t in texts if t), meta

        start = time.perf_counter()
        try:
            text, meta = await asyncio.wait_for(
                retry_on_connect_error(send, name=label, retries=settings.A2A_CONNECT_RETRIES,
                                       backoff=settings.A2A_RETRY_BACKOFF_SECONDS),
                timeout=settings.A2A_TIMEOUT_SECONDS,
            )
        except Exception as e:
            status = "timeout" if isinstance(e, asyncio.TimeoutError) else "error"
            agent.breaker.record_failure()
            CLIENT_CALLS.labels(label, status).inc()
            logger.warning("a2a_call_failed agent=%s status=%s error=%r", label, status, e,
                           extra={"remote_agent": label, "status": status})
            return UNAVAILABLE_NOTE.format(name=agent.card.name), {"agent": label, "status": status}
        finally:
            CLIENT_LATENCY.labels(label).observe(time.perf_counter() - start)

        # The expert answered, but with a fallback ("sorry") reply: count it against the circuit
        # too, so an expert that keeps failing (e.g. out of LLM quota) stops slowing every chat.
        if meta.get("degraded") and meta.get("status") in ("error", "timeout"):
            agent.breaker.record_failure()
        else:
            agent.breaker.record_success()
        CLIENT_CALLS.labels(label, "ok").inc()
        return text or "The agent returned no answer.", meta

    @staticmethod
    def _build_message(agent: RemoteAgent, question: str, context: OrchestratorContext) -> Message:
        parts = [new_text_part(question)]
        if context.image_path and agent.accepts_images:
            path = Path(context.image_path)
            media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            parts.append(new_raw_part(path.read_bytes(), media_type=media_type, filename=path.name))
        metadata = Struct()
        metadata.update({"user_id": context.user_id, "trace_id": context.trace_id})
        # Distributed tracing: when LangSmith is tracing this chat, send the current run's
        # headers along, so the expert's steps (CrewAI, LangChain or LangGraph) appear nested
        # under this tool call in one LangSmith trace.
        run = get_current_run_tree()
        if run is not None:
            metadata.update({f"ls_{k}": v for k, v in run.to_headers().items()})
        return Message(message_id=str(uuid.uuid4()), role=Role.ROLE_USER, parts=parts, metadata=metadata)


def a2a_tool(registry: AgentRegistry, agent: RemoteAgent) -> BaseTool:
    """One expert as a LangChain tool. The LLM only writes the question; the farmer's id, trace id
    and photo come from the run's context (`runtime`), which the LLM never sees.

    response_format="content_and_artifact": the LLM sees only the answer text, while the reply
    metadata (tools used, latency, framework) is kept on the ToolMessage for metrics and evals.
    """
    skills = "; ".join(f"{s.name}: {s.description}" for s in agent.card.skills)

    async def ask(question: str, runtime: ToolRuntime[OrchestratorContext]):
        return await registry.call(agent, question, runtime.context)

    return tool(
        agent.tool_name,
        description=f"{agent.card.description} Skills - {skills}",
        args_schema=AgentQuestion,
        response_format="content_and_artifact",
    )(ask)
