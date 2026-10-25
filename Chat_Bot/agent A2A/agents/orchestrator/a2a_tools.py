"""Turn remote A2A agents into LangChain tools for the orchestrator.

discover_agents() reads each URL's Agent Card; a2a_tool() wraps one remote agent as a tool whose
name and description come from its card. Calling the tool sends an A2A message to that agent.
"""
import mimetypes
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List

import httpx
from a2a.client import A2ACardResolver, Client, ClientConfig, ClientFactory
from a2a.helpers import get_stream_response_text, new_raw_part, new_text_part
from a2a.types import AgentCard, Message, Role, SendMessageRequest
from google.protobuf.struct_pb2 import Struct
from langchain.tools import BaseTool, ToolRuntime, tool
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, Field

from agent.state.agent_state import ChatContext


class AgentQuestion(BaseModel):
    question: str = Field(
        description="The farmer's question for this agent, rewritten so it makes sense on its own."
    )


@dataclass
class RemoteAgent:
    """One remote A2A agent: its card (what it does) and a client to talk to it."""

    url: str
    card: AgentCard
    client: Client

    @property
    def intent(self) -> str:
        """The intent label - by project convention, the id of the card's first skill."""
        return self.card.skills[0].id if self.card.skills else self.tool_name

    @property
    def tool_name(self) -> str:
        """e.g. "Tea Disease Agent" -> "ask_tea_disease_agent"."""
        return "ask_" + re.sub(r"[^a-z0-9]+", "_", self.card.name.lower()).strip("_")

    @property
    def accepts_images(self) -> bool:
        return any(mode.startswith("image/") for mode in self.card.default_input_modes)


async def discover_agents(urls: List[str], http_client: httpx.AsyncClient) -> List[RemoteAgent]:
    """Fetch each agent's card and build a client for it. Unreachable agents are skipped."""
    factory = ClientFactory(ClientConfig(httpx_client=http_client, streaming=False))
    agents = []
    for url in urls:
        try:
            card = await A2ACardResolver(http_client, url).get_agent_card()
        except Exception as e:
            print(f"A2A agent at {url} is not reachable, skipping it: {e}")
            continue
        agents.append(RemoteAgent(url=url, card=card, client=factory.create(card)))
        print(f"A2A agent found: {card.name} ({url})")
    return agents


def a2a_tool(agent: RemoteAgent) -> BaseTool:
    """Wrap one remote agent as a tool. The LLM only writes the question; the farmer's user_id
    and photo come from the ChatContext (via `runtime`, which the LLM never sees)."""
    skills = "; ".join(f"{s.name}: {s.description}" for s in agent.card.skills)
    description = f"{agent.card.description} Skills - {skills}"

    async def ask(question: str, runtime: ToolRuntime[ChatContext]) -> str:
        context = runtime.context
        parts = [new_text_part(question)]
        if context.image_path and agent.accepts_images:
            path = Path(context.image_path)
            media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            parts.append(new_raw_part(path.read_bytes(), media_type=media_type, filename=path.name))

        metadata = Struct()
        metadata.update({"user_id": context.user_id})
        # Distributed tracing: hand the remote agent this tool call's LangSmith run, so its own
        # steps appear *inside* this trace instead of as a separate one. (No-op if tracing is off.)
        run = get_current_run_tree()
        if run is not None:
            metadata.update({f"ls_{k}": v for k, v in run.to_headers().items()})
        message = Message(
            message_id=str(uuid.uuid4()), role=Role.ROLE_USER, parts=parts, metadata=metadata
        )

        texts = []
        async for response in agent.client.send_message(SendMessageRequest(message=message)):
            texts.append(get_stream_response_text(response))
        return "\n".join(t for t in texts if t) or "The agent returned no answer."

    return tool(
        agent.tool_name, description=description, args_schema=AgentQuestion
    )(ask)
