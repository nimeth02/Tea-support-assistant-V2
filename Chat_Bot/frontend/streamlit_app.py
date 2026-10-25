"""Kept so the old command still works: runs the LangGraph-router UI (app_agent.py).

Each agent variant now has its own UI - app_agent.py, app_langchain.py, app_a2a.py,
app_crewai.py. Start one together with its backend via:  python launch.py up <variant>
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("app_agent.py")), run_name="__main__")
