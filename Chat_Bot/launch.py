"""Start any of the four agent variants - backend, UI, or both. Run from the Chat_Bot folder.

    python launch.py up agent          # backend + UI for the LangGraph router
    python launch.py up langchain      # ... the LangChain supervisor
    python launch.py up a2a            # ... the A2A network (also starts the 3 A2A expert servers)
    python launch.py up crewai         # ... the CrewAI crew
    python launch.py up all            # all four at once

    python launch.py backend <key>     # only the FastAPI backend
    python launch.py ui <key>          # only the Streamlit UI
    python launch.py a2a-servers       # only the A2A expert servers (ports 9101-9103)
    python launch.py list              # show ports and LangSmith projects

Each variant is its own set of processes (see variants.py for why). Ctrl+C stops everything
that `up` started.
"""
import argparse
import os
import subprocess
import sys
import time
from typing import List

from variants import ROOT, VARIANTS, Variant, get_variant


def _env(variant: Variant) -> dict:
    return {**os.environ, "AGENT_VARIANT": variant.key, "TEA_API_URL": variant.backend_url,
            "PYTHONIOENCODING": "utf-8"}


def backend_cmd(variant: Variant) -> List[str]:
    return [sys.executable, "-m", "uvicorn", "backend.main:app",
            "--host", "127.0.0.1", "--port", str(variant.backend_port)]


def ui_cmd(variant: Variant) -> List[str]:
    return [sys.executable, "-m", "streamlit", "run", str(ROOT / "frontend" / f"app_{variant.key}.py"),
            "--server.port", str(variant.ui_port), "--server.headless", "true",
            "--theme.primaryColor", variant.color]


def a2a_servers_cmd() -> List[str]:
    return [sys.executable, "-c",
            "from agent_loader import mount_agent_variant; mount_agent_variant('a2a'); "
            "import asyncio; from agent.a2a_agents.run_agents import main; asyncio.run(main())"]


def processes_for(variant: Variant) -> List[tuple]:
    procs = []
    if variant.key == "a2a":
        procs.append((f"{variant.key}:a2a-servers", a2a_servers_cmd()))
    if variant.key == "hybrid":  # its 3 experts (CrewAI, LangChain, LangGraph), one process each
        procs.append((f"{variant.key}:a2a-servers",
                      [sys.executable, str(variant.path / "run_hybrid.py"), "experts"]))
    procs.append((f"{variant.key}:backend", backend_cmd(variant)))
    procs.append((f"{variant.key}:ui", ui_cmd(variant)))
    return procs


def run_all(jobs: List[tuple]) -> None:
    """Start every (name, cmd, variant) job and wait; Ctrl+C or any exit stops the rest."""
    running = []
    for name, cmd, variant in jobs:
        print(f"[launch] starting {name}")
        running.append((name, subprocess.Popen(cmd, cwd=ROOT, env=_env(variant))))
        if name.endswith("a2a-servers"):
            time.sleep(8)  # let the expert servers come up before the backend looks for them
    try:
        while True:
            for name, proc in running:
                if proc.poll() is not None:
                    print(f"[launch] {name} exited with code {proc.returncode} - stopping the rest")
                    return
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for _, proc in running:
            if proc.poll() is None:
                proc.terminate()


def print_list() -> None:
    print(f"{'key':10} {'backend':24} {'ui':24} langsmith project")
    for v in VARIANTS.values():
        print(f"{v.key:10} {v.backend_url:24} {'http://localhost:' + str(v.ui_port):24} {v.langsmith_project}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["up", "backend", "ui", "a2a-servers", "list"])
    parser.add_argument("variant", nargs="?", default="agent", help=f"one of {', '.join(VARIANTS)} (or 'all' with up)")
    args = parser.parse_args()

    if args.command == "list":
        return print_list()
    if args.command == "a2a-servers":
        return run_all([("a2a:a2a-servers", a2a_servers_cmd(), VARIANTS["a2a"])])

    variants = list(VARIANTS.values()) if args.variant == "all" and args.command == "up" else [get_variant(args.variant)]
    jobs = []
    for v in variants:
        if args.command == "up":
            jobs += [(name, cmd, v) for name, cmd in processes_for(v)]
        elif args.command == "backend":
            jobs.append((f"{v.key}:backend", backend_cmd(v), v))
        else:
            jobs.append((f"{v.key}:ui", ui_cmd(v), v))
    for v in variants:
        print(f"[launch] {v.title}: UI http://localhost:{v.ui_port}  API {v.backend_url}  "
              f"LangSmith project '{v.langsmith_project}'")
    run_all(jobs)


if __name__ == "__main__":
    main()
