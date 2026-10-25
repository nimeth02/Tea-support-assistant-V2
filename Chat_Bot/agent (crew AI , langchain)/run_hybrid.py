"""Run this variant's own processes straight from its folder (variant key: "hybrid").

All code here imports itself as `agent...`, like every other variant. This script makes this
folder act as the `agent` package (through Chat_Bot/agent_loader.py when it is there, so the
LangSmith project etc. match the rest of the project), then runs the command.
Run from the Chat_Bot folder:

    python "agent (crew AI , langchain)/run_hybrid.py" experts                  # all 3 A2A experts
    python "agent (crew AI , langchain)/run_hybrid.py" experts --agent disease  # one expert
    python "agent (crew AI , langchain)/run_hybrid.py" smoke [--ask]            # deploy check
    python "agent (crew AI , langchain)/run_hybrid.py" eval [--limit 5]         # quality gate
    python "agent (crew AI , langchain)/run_hybrid.py" ingest --reset           # build Chroma KBs

Backend + UI (and the experts too) are started by the project launcher:
    python launch.py up hybrid
"""
import importlib.util
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAT_BOT = HERE.parent
VARIANT_KEY = "hybrid"

COMMANDS = {
    "experts": "agent.a2a_servers.run_agents",
    "smoke": "agent.ops.smoke_test",
    "eval": "agent.evals.run_eval",
    "ingest": "agent.db.vector.chroma.ingest_chroma",
}


def register_as_agent_package() -> None:
    """Make `import agent` load this folder, whatever its name on disk."""
    if "agent" in sys.modules:
        return
    if str(CHAT_BOT) not in sys.path:
        sys.path.insert(0, str(CHAT_BOT))  # backend/, observability/, agent_loader.py live there
    try:
        from agent_loader import mount_agent_variant
        from variants import VARIANTS
        if VARIANT_KEY in VARIANTS:
            mount_agent_variant(VARIANT_KEY)
            return
    except ImportError:
        pass  # no project launcher (e.g. a slim checkout): mount the folder ourselves
    spec = importlib.util.spec_from_file_location(
        "agent", HERE / "__init__.py", submodule_search_locations=[str(HERE)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent"] = module
    spec.loader.exec_module(module)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(2)
    command, rest = sys.argv[1], sys.argv[2:]
    register_as_agent_package()
    sys.argv = [COMMANDS[command], *rest]
    runpy.run_module(COMMANDS[command], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
