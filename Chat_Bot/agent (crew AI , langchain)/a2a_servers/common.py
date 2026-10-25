"""Turn any expert (CrewAI, LangChain or LangGraph) into a production A2A server.

A2A in one paragraph: every agent is a small web server. It publishes an *Agent Card* (a JSON
description of who it is and what skills it has) at /.well-known/agent-card.json, and accepts
messages over JSON-RPC at /a2a. A message is a list of *parts* (text, file bytes such as the leaf
photo) plus *metadata*. Any A2A client can discover the agent from its card and talk to it,
whatever framework either side is written in. That is why the disease expert can be CrewAI and
the fertilizer expert LangChain: they only meet at this protocol.

What every server gets from this module (the AIOps part):
  /.well-known/agent-card.json  public card, lists the auth it needs
  /a2a                          JSON-RPC endpoint, protected by a bearer token
  /healthz                      liveness: the process is up
  /readyz                       readiness: LLM key, model files, databases are usable (503 if not)
  /metrics                      Prometheus metrics (requests, latency, tools, fallbacks, in-flight)
  JSON logs carrying the caller's trace id, a per-question timeout, input validation, and a
  reply whose metadata says which tools ran, how long it took, and whether it was a fallback.
  LangSmith distributed tracing: the expert's steps nest under the caller's trace.

Message conventions (the orchestrator relies on them):
  - the card's first skill id is the intent label ("disease", "fertilize", "company_help")
  - request metadata: user_id (int), trace_id (str), ls_* (LangSmith trace headers, optional)
  - a leaf photo is a raw part with an image media type, sent only to agents whose card lists
    an image type in default_input_modes
"""
import asyncio
import hmac
import logging
import time
import uuid
from pathlib import Path
from typing import Callable, List, Optional

from a2a.helpers import new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import add_a2a_routes_to_fastapi, create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities, AgentCard, AgentInterface, AgentProvider, AgentSkill, HTTPAuthSecurityScheme,
    Message, Role, SecurityRequirement, SecurityScheme, StringList,
)
from a2a.utils.constants import TransportProtocol
from a2a.utils.errors import UnsupportedOperationError
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google.protobuf.struct_pb2 import Struct
from langsmith import run_helpers as ls_rh
from prometheus_client import make_asgi_app

from agent.agents.base import Expert, ExpertAnswer
from agent.config import settings
from agent.ops import guardrails
from agent.ops.health import Check, run_checks
from agent.ops.telemetry import (
    EXPERT_IN_FLIGHT, EXPERT_LATENCY, EXPERT_REQUESTS, EXPERT_TOOL_CALLS, set_agent_name, set_trace_id,
)

logger = logging.getLogger("agent.a2a")

A2A_PROTOCOL_VERSION = "1.0"
RPC_PATH = "/a2a"
AUTH_SCHEME_NAME = "bearer"
UPLOAD_DIR = Path("data/uploads/a2a")  # photos received over A2A, deleted after the answer
IMAGE_SUFFIXES = {"image/jpeg": ".jpg", "image/png": ".png"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024


# ---- Agent Card ------------------------------------------------------------------------------

def build_agent_card(name: str, description: str, url: str, skills: List[AgentSkill],
                     framework: str, input_modes: Optional[List[str]] = None) -> AgentCard:
    """The card this server publishes, so clients know what it does and how to reach it."""
    card = AgentCard(
        name=name,
        description=description,
        version="2.0.0",
        provider=AgentProvider(organization=f"Tea Support Assistant ({framework})", url=url),
        supported_interfaces=[AgentInterface(
            url=url.rstrip("/") + RPC_PATH,
            protocol_binding=TransportProtocol.JSONRPC.value,
            protocol_version=A2A_PROTOCOL_VERSION,
        )],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=input_modes or ["text/plain"],
        default_output_modes=["text/plain"],
        skills=skills,
    )
    if settings.A2A_AUTH_TOKEN:
        # Tell clients (in a standard way) that the endpoint needs "Authorization: Bearer ...".
        card.security_schemes[AUTH_SCHEME_NAME].CopyFrom(SecurityScheme(
            http_auth_security_scheme=HTTPAuthSecurityScheme(
                scheme="bearer", description="Shared service token (A2A_AUTH_TOKEN)"
            )
        ))
        card.security_requirements.append(
            SecurityRequirement(schemes={AUTH_SCHEME_NAME: StringList(list=[])})
        )
    return card


# ---- Executor: one A2A message in, one answer out ---------------------------------------------

class ExpertExecutor(AgentExecutor):
    """Runs one expert behind A2A. The expert is built lazily on the first question, so a missing
    API key shows up as "not ready" on /readyz instead of a crash at startup."""

    def __init__(self, expert_factory: Callable[[], Expert], name: str, framework: str):
        self._factory = expert_factory
        self._expert: Optional[Expert] = None
        self._lock = asyncio.Lock()
        self.name = name
        self.framework = framework

    async def _get_expert(self) -> Expert:
        async with self._lock:
            if self._expert is None:
                # Building an agent loads models/DB clients; keep it off the event loop.
                self._expert = await asyncio.to_thread(self._factory)
        return self._expert

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        message = context.message
        metadata = dict(message.metadata) if message else {}
        trace_id = set_trace_id(str(metadata.get("trace_id") or ""))
        set_agent_name(f"{self.name}-agent")

        question = context.get_user_input()
        # Metadata numbers arrive as floats (protobuf Struct), so convert back to int.
        user_id = int(metadata.get("user_id") or 0)
        image_path = self._save_image(message)

        start = time.perf_counter()
        status = "ok"
        EXPERT_IN_FLIGHT.labels(self.name).inc()
        try:
            check = guardrails.check_input(question)
            if not check.allowed or user_id <= 0:
                status = "rejected"
                answer = ExpertAnswer(
                    text=check.message or "A valid farmer id is required for this question.",
                    degraded=True,
                )
            else:
                expert = await self._get_expert()
                # Continue the caller's LangSmith trace (headers arrive as "ls_*" metadata), so
                # this expert's steps nest under the orchestrator's tool call. No headers = no-op.
                trace_headers = {k[3:]: str(v) for k, v in metadata.items() if k.startswith("ls_")}
                with ls_rh.tracing_context(parent=trace_headers or None):
                    answer = await asyncio.wait_for(
                        expert.answer(check.text, user_id, image_path), timeout=settings.EXPERT_TIMEOUT_SECONDS
                    )
                if answer.degraded:
                    status = "error"
        except asyncio.TimeoutError:
            status = "timeout"
            answer = ExpertAnswer(text="Sorry, this took too long. Please try again.", degraded=True)
        except Exception:
            status = "error"
            logger.exception("expert_failed agent=%s", self.name)
            answer = ExpertAnswer(text="Sorry, something went wrong on my side. Please try again.", degraded=True)
        finally:
            EXPERT_IN_FLIGHT.labels(self.name).dec()
            if image_path:
                Path(image_path).unlink(missing_ok=True)

        latency = time.perf_counter() - start
        EXPERT_REQUESTS.labels(self.name, self.framework, status).inc()
        EXPERT_LATENCY.labels(self.name, self.framework).observe(latency)
        for tool in answer.tools_used:
            EXPERT_TOOL_CALLS.labels(self.name, tool).inc()
        logger.info(
            "expert_answer agent=%s status=%s latency_ms=%.0f tools=%s", self.name, status,
            latency * 1000, answer.tools_used,
            extra={"agent": self.name, "framework": self.framework, "status": status,
                   "latency_ms": round(latency * 1000), "tools": answer.tools_used, "user_id": user_id},
        )

        reply_metadata = Struct()
        reply_metadata.update({
            "trace_id": trace_id,
            "agent": self.name,
            "framework": self.framework,
            "status": status,
            "latency_ms": round(latency * 1000),
            "tools_used": answer.tools_used,
            "model": answer.model or "",
            "degraded": answer.degraded,
        })
        await event_queue.enqueue_event(Message(
            message_id=str(uuid.uuid4()),
            role=Role.ROLE_AGENT,
            parts=[new_text_part(guardrails.clean_output(answer.text))],
            context_id=context.context_id,
            task_id=context.task_id,
            metadata=reply_metadata,
        ))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError("This agent answers in one step; there is nothing to cancel.")

    @staticmethod
    def _save_image(message: Optional[Message]) -> Optional[str]:
        for part in message.parts if message else []:
            suffix = IMAGE_SUFFIXES.get(part.media_type)
            if part.HasField("raw") and suffix and 0 < len(part.raw) <= MAX_IMAGE_BYTES:
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
                path.write_bytes(part.raw)
                return str(path)
        return None


# ---- The web app -----------------------------------------------------------------------------

def build_a2a_app(card: AgentCard, executor: AgentExecutor, readiness_checks: List[Check]) -> FastAPI:
    if settings.is_production and not settings.A2A_AUTH_TOKEN:
        raise RuntimeError("A2A_AUTH_TOKEN must be set when ENVIRONMENT is staging/production.")
    if not settings.A2A_AUTH_TOKEN:
        logger.warning("a2a_auth_disabled agent=%s (set A2A_AUTH_TOKEN outside dev)", card.name)

    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card)
    app = FastAPI(title=card.name, description=card.description, version=card.version)
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url=RPC_PATH),
    )

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        """Only the A2A endpoint needs the token; the card and health endpoints stay public."""
        if settings.A2A_AUTH_TOKEN and request.url.path.startswith(RPC_PATH):
            sent = request.headers.get("authorization", "")
            expected = f"Bearer {settings.A2A_AUTH_TOKEN}"
            if not hmac.compare_digest(sent.encode(), expected.encode()):  # constant-time compare
                EXPERT_REQUESTS.labels(getattr(executor, "name", card.name), getattr(executor, "framework", "-"),
                                       "unauthorized").inc()
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    @app.get("/healthz", tags=["ops"])
    async def healthz():
        return {"status": "ok", "agent": card.name}

    @app.get("/readyz", tags=["ops"])
    async def readyz():
        ok, results = await asyncio.to_thread(run_checks, readiness_checks)
        return JSONResponse({"ready": ok, "checks": results}, status_code=200 if ok else 503)

    app.mount("/metrics", make_asgi_app())
    return app
