"""LangSmith tracing for one chat request - the same for every agent variant.

LangSmith (https://smith.langchain.com) is a hosted trace viewer: each chat becomes a tree of
steps (LLM calls, tool calls, agent steps) you can click through.

chat_run() opens a *root run* for the request. Anything LangChain does inside it nests under
it automatically, because LangChain looks up "the current LangSmith run" from context. CrewAI
does not use LangChain, so langsmith_crewai.py feeds its steps into the same tree.

Each variant sends to its own project (see variants.py), so the four never mix.
Tracing only happens when LANGSMITH_TRACING=true and LANGSMITH_API_KEY is set in .env.
"""
import logging
import uuid
from contextlib import contextmanager
from typing import Iterator, Optional

import langsmith as ls
from langsmith import utils as ls_utils

from variants import Variant

logger = logging.getLogger("observability")

_project_urls: dict = {}


def tracing_enabled() -> bool:
    return bool(ls_utils.tracing_is_enabled())


def project_url(variant: Variant) -> Optional[str]:
    """Web link to the variant's LangSmith project, or None (tracing off / no runs sent yet)."""
    if not tracing_enabled():
        return None
    name = variant.langsmith_project
    if name not in _project_urls:
        try:
            _project_urls[name] = ls.Client().read_project(project_name=name).url
        except Exception as e:  # the project is created by the first trace, so it may not exist yet
            logger.info("LangSmith project %s not available yet: %s", name, e)
            return None
    return _project_urls[name]


def run_url(variant: Variant, run_id: uuid.UUID) -> Optional[str]:
    base = project_url(variant)
    return f"{base}/r/{run_id}?poll=true" if base else None


@contextmanager
def chat_run(variant: Variant, trace_id: str, message: str, user_id: int,
             image_path: Optional[str]) -> Iterator[Optional[ls.RunTree]]:
    """Root LangSmith run for one chat. Yields the run (or None when tracing is off).

    The backend's own trace id goes into the metadata, so a LangSmith run can be matched to
    the same request on the local Observability page.
    """
    if not tracing_enabled():
        yield None
        return
    with ls.trace(
        name=f"chat:{variant.key}",
        run_type="chain",
        inputs={"message": message, "user_id": user_id, "has_image": bool(image_path)},
        project_name=variant.langsmith_project,
        run_id=uuid.uuid4(),
        tags=[f"variant:{variant.key}"],
        metadata={"variant": variant.key, "framework": variant.framework, "trace_id": trace_id},
    ) as run:
        yield run
