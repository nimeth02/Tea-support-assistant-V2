"""Shared plumbing for exposing an expert agent as an A2A (Agent2Agent protocol) server.

A2A in one paragraph: every agent is a small web server. It publishes an *Agent Card* (a JSON
description of who it is and what skills it has) at /.well-known/agent-card.json, and accepts
messages over JSON-RPC (`SendMessage`). A message is a list of *parts* - text, raw file bytes
(our leaf photo), or data - plus optional metadata. Any A2A client can discover the agent from
its card and talk to it, whatever framework either side is written in.

Conventions used by this project (the orchestrator relies on them):
  - The first skill's `id` is the intent label ("disease", "fertilize", "company_help").
  - The farmer's id travels in the message metadata as `user_id`.
  - A leaf photo travels as a raw part with an image media type, and only to agents whose card
    lists an image type in `default_input_modes`.
"""
import uuid
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from a2a.helpers import new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import add_a2a_routes_to_fastapi, create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.utils.constants import TransportProtocol
from a2a.utils.errors import UnsupportedOperationError
from fastapi import FastAPI
from langsmith import run_helpers as ls_rh

A2A_PROTOCOL_VERSION = "1.0"
RPC_PATH = "/a2a"  # JSON-RPC endpoint; the card is served at /.well-known/agent-card.json
UPLOAD_DIR = Path("data/uploads/a2a")  # photos received over A2A, deleted after the answer
IMAGE_SUFFIXES = {"image/jpeg": ".jpg", "image/png": ".png"}


def build_agent_card(
    name: str,
    description: str,
    url: str,
    skills: List[AgentSkill],
    input_modes: Optional[List[str]] = None,
) -> AgentCard:
    """The Agent Card this server publishes, so clients know what it does and how to reach it."""
    return AgentCard(
        name=name,
        description=description,
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(
                url=url.rstrip("/") + RPC_PATH,
                protocol_binding=TransportProtocol.JSONRPC.value,
                protocol_version=A2A_PROTOCOL_VERSION,
            )
        ],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=input_modes or ["text/plain"],
        default_output_modes=["text/plain"],
        skills=skills,
    )


def build_a2a_app(card: AgentCard, executor: AgentExecutor) -> FastAPI:
    """A FastAPI app that serves the Agent Card and the A2A JSON-RPC endpoint for one agent."""
    handler = DefaultRequestHandler(
        agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card
    )
    app = FastAPI(title=card.name, description=card.description)
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url=RPC_PATH),
    )
    return app


@dataclass
class ExpertRequest:
    """What an expert needs from an incoming A2A message."""

    question: str
    user_id: int
    image_path: Optional[str] = None  # local copy of an attached photo, if any
    # LangSmith trace headers from the caller ("langsmith-trace", "baggage"), if it is tracing
    trace_headers: Optional[Dict[str, str]] = None


class ExpertExecutor(AgentExecutor):
    """Base A2A executor: unpack the message, let the expert answer, reply with one text message.

    Subclasses only implement `answer()`. A single Message reply is the A2A "immediate
    response" pattern - fine here because each expert answers in one go.
    """

    @abstractmethod
    async def answer(self, request: ExpertRequest) -> str:
        """Return the expert's answer to one question."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        request = self._unpack(context)
        try:
            # Continue the caller's LangSmith trace: this agent's LLM/tool calls nest under the
            # orchestrator's tool call. parent=None (no headers) leaves tracing as it is.
            with ls_rh.tracing_context(parent=request.trace_headers or None):
                answer = await self.answer(request)
        finally:
            if request.image_path:
                Path(request.image_path).unlink(missing_ok=True)
        await event_queue.enqueue_event(
            new_text_message(answer, context_id=context.context_id, task_id=context.task_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError("This agent answers in one step; there is nothing to cancel.")

    @staticmethod
    def _unpack(context: RequestContext) -> ExpertRequest:
        message = context.message
        metadata = dict(message.metadata) if message else {}

        image_path = None
        for part in message.parts if message else []:
            suffix = IMAGE_SUFFIXES.get(part.media_type)
            if part.HasField("raw") and suffix:
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
                path.write_bytes(part.raw)
                image_path = str(path)
                break

        return ExpertRequest(
            question=context.get_user_input(),
            # Metadata numbers arrive as floats (protobuf Struct), so convert back to int.
            user_id=int(metadata.get("user_id", 0)),
            image_path=image_path,
            trace_headers={k[3:]: str(v) for k, v in metadata.items() if k.startswith("ls_")} or None,
        )
