"""Start all expert A2A servers in one process (one port each).

Run from the Chat_Bot folder, before starting the backend:

    python -m agent.a2a_agents.run_agents

Each one is an independent A2A agent - in production they could just as well run on different
machines; the orchestrator only needs their URLs (see A2A_* settings in agent/config.py).
"""
import asyncio
from urllib.parse import urlparse

import uvicorn

from agent.a2a_agents import disease_server, fertilize_server, help_server
from agent.config import settings

SERVERS = [
    (disease_server, settings.A2A_DISEASE_AGENT_URL),
    (fertilize_server, settings.A2A_FERTILIZE_AGENT_URL),
    (help_server, settings.A2A_HELP_AGENT_URL),
]


async def main() -> None:
    servers = []
    for module, url in SERVERS:
        parsed = urlparse(url)
        config = uvicorn.Config(module.create_app(), host=parsed.hostname, port=parsed.port, log_level="info")
        servers.append(uvicorn.Server(config))
        print(f"A2A agent card: {url}/.well-known/agent-card.json")
    await asyncio.gather(*(server.serve() for server in servers))


if __name__ == "__main__":
    asyncio.run(main())
