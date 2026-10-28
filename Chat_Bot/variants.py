"""The versions ("variants") of the tea agent, and where each one runs.

Every variant folder imports itself as the `agent` package (e.g. `from agent.config import ...`),
so two variants cannot live in the same Python process. Each one therefore gets its own
backend process (own port), its own Streamlit UI (own port), its own LangSmith project and its
own observability database. agent_loader.py makes the chosen folder act as `agent`.

Start one with:   python launch.py up <key>      (keys: agent, langchain, a2a, crewai, hybrid, voice)
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Variant:
    key: str            # short name used on the command line and in AGENT_VARIANT
    folder: str         # folder under Chat_Bot/ that holds this version of the agent
    title: str          # shown in the UI
    framework: str      # one-line "what is this built with"
    backend_port: int
    ui_port: int
    color: str          # UI accent colour
    default_project: str  # LangSmith project name (override with LANGSMITH_PROJECT_<KEY>)

    @property
    def path(self) -> Path:
        return ROOT / self.folder

    @property
    def backend_url(self) -> str:
        return f"http://127.0.0.1:{self.backend_port}"

    @property
    def langsmith_project(self) -> str:
        return os.getenv(f"LANGSMITH_PROJECT_{self.key.upper()}", self.default_project)


VARIANTS: Dict[str, Variant] = {
    v.key: v
    for v in [
        Variant("agent", "agent", "LangGraph router",
                "LangGraph state graph: classify intent, then route to one expert",
                8001, 8510, "#2e7d32", "tea-support-agent-langgraph"),
        Variant("langchain", "agent (langchain)", "LangChain supervisor",
                "LangChain create_agent supervisor that calls experts as tools",
                8002, 8511, "#1565c0", "tea-support-agent-langchain"),
        Variant("a2a", "agent A2A", "A2A agent network",
                "Supervisor that talks to expert agents over the A2A protocol (HTTP)",
                8003, 8512, "#6a1b9a", "tea-support-agent-a2a"),
        Variant("crewai", "agent (crewAI)", "CrewAI crew",
                "CrewAI hierarchical crew: a manager delegates to expert agents",
                8004, 8513, "#e65100", "tea-support-agent-crewai"),
        Variant("hybrid", "agent (crew AI , langchain)", "Multi-framework A2A",
                "LangChain supervisor + CrewAI, LangChain and LangGraph experts over A2A (ports 9201-9203)",
                8005, 8514, "#00695c", "tea-support-agent-hybrid"),
        Variant("voice", "agent multilangual and voice also", "Multilingual + voice",
                "LangGraph router that understands and answers in Sinhala, Tamil or English, by text or voice",
                8006, 8515, "#ad1457", "tea-support-agent-voice"),
    ]
}

DEFAULT_VARIANT = "agent"


def get_variant(key: str = None) -> Variant:
    """The variant named by `key`, or by the AGENT_VARIANT env var (default: "agent")."""
    key = (key or os.getenv("AGENT_VARIANT") or DEFAULT_VARIANT).lower()
    if key not in VARIANTS:
        raise ValueError(f"Unknown agent variant {key!r} - choose one of {', '.join(VARIANTS)}")
    return VARIANTS[key]
