"""Start the expert A2A servers - one process per expert, like in production.

    python "agent (crew AI , langchain)/run_hybrid.py" experts                  # all three
    python "agent (crew AI , langchain)/run_hybrid.py" experts --agent disease  # just one

Why one process each, even on a laptop:
  * isolation - a crash or memory leak in one expert (say, torch in the disease expert) can't
    take the other two down, which is the whole point of splitting them into A2A services;
  * honest metrics - each expert's /metrics shows only its own numbers;
  * no cross-talk - CrewAI and LangChain both hook into LangSmith tracing, and sharing one
    process made their tracers trip over each other.
With several agents requested, this script starts one child per agent and stops them all on
Ctrl+C. In Docker/Kubernetes each container runs exactly one (`--agent <name>`).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import uvicorn

from agent.config import settings
from agent.ops.log_config import setup_logging
from agent.ops.telemetry import set_agent_name

FRAMEWORKS = {"disease": "crewai", "fertilize": "langchain", "help": "langgraph"}
LAUNCHER = Path(__file__).resolve().parents[1] / "run_hybrid.py"


def serve_one(name: str) -> None:
    from importlib import import_module

    module = import_module(f"agent.a2a_servers.{name}_server")
    set_agent_name(f"{name}-agent")
    setup_logging(service=f"{name}-agent")
    url = urlparse(settings.expert_urls[name])
    print(f"{name}-agent ({FRAMEWORKS[name]}): card at {settings.expert_urls[name]}/.well-known/agent-card.json",
          flush=True)
    uvicorn.run(
        module.create_app(), host=settings.A2A_BIND_HOST or url.hostname, port=url.port,
        log_config=None,               # keep our JSON logging instead of uvicorn's own format
        timeout_graceful_shutdown=30,  # let in-flight questions finish on SIGTERM
    )


def supervise(names: list) -> None:
    """Start one child process per expert; if one dies, stop the rest (so it gets noticed)."""
    children = {n: subprocess.Popen([sys.executable, str(LAUNCHER), "experts", "--agent", n]) for n in names}
    try:
        while True:
            for name, proc in children.items():
                if proc.poll() is not None:
                    print(f"[experts] {name}-agent exited with code {proc.returncode} - stopping the rest")
                    return
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for proc in children.values():
            if proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=list(FRAMEWORKS), action="append",
                        help="which expert to run (repeatable); default: all, one process each")
    args = parser.parse_args()
    names = args.agent or list(FRAMEWORKS)
    if len(names) == 1:
        serve_one(names[0])
    else:
        supervise(names)
