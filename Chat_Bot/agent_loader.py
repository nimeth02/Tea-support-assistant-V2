"""Make one variant folder importable as the `agent` package.

Why: every variant's code says `from agent.xxx import ...`, but only the "agent" folder is
actually called `agent`. Folder names like "agent (crewAI)" are not valid Python names, so we
register the chosen folder under the name `agent` by hand. After mount_agent_variant() runs,
`import agent.config` loads <variant folder>/config.py.

It must run before anything imports `agent` - the backend calls it first thing.
"""
import importlib.util
import os
import sys

from dotenv import load_dotenv

from variants import ROOT, Variant, get_variant

_mounted: Variant = None


def mount_agent_variant(key: str = None) -> Variant:
    """Register the variant's folder as the `agent` package and set its LangSmith project."""
    global _mounted
    variant = get_variant(key)
    if _mounted is not None:
        if _mounted.key != variant.key:
            raise RuntimeError(f"Variant {_mounted.key!r} is already loaded in this process")
        return _mounted

    # Load .env into os.environ first, so LANGSMITH_* settings are visible to the langsmith SDK
    # (and to LANGSMITH_PROJECT_<KEY> overrides) before any agent code runs.
    load_dotenv(ROOT / ".env", override=False)
    os.environ["AGENT_VARIANT"] = variant.key
    # Real env vars beat .env values in pydantic-settings, so agent/config.py picks this up too.
    os.environ["LANGSMITH_PROJECT"] = variant.langsmith_project

    if "agent" in sys.modules:
        raise RuntimeError("`agent` was imported before mount_agent_variant() was called")

    if variant.folder != "agent":  # the plain "agent" folder already imports normally
        spec = importlib.util.spec_from_file_location(
            "agent", variant.path / "__init__.py", submodule_search_locations=[str(variant.path)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["agent"] = module
        spec.loader.exec_module(module)

    _mounted = variant
    return variant


def active_variant() -> Variant:
    """The variant mounted in this process (mounting the default one if none was yet)."""
    return _mounted or mount_agent_variant()
