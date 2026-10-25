"""Test setup: load this folder as the `agent` package and give async tests an event loop.

Run from the Chat_Bot folder:
    .venv\\Scripts\\python -m pytest "agent (crew AI , langchain)/tests" -q

No test here calls a real LLM, so they are free, fast and safe to run in CI on every push.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent))  # Chat_Bot/, for backend/, observability/ and agent_loader

# Load run_hybrid.py by path (its folder name has spaces, so it can't be imported normally).
_spec = importlib.util.spec_from_file_location("run_hybrid", HERE / "run_hybrid.py")
run_hybrid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_hybrid)
run_hybrid.register_as_agent_package()


@pytest.fixture
def anyio_backend():
    return "asyncio"
