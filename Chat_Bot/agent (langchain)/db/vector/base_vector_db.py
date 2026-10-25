"""Common interface for the vector databases the RAG tools query.

Every implementation returns the same shape from ``query`` so agents and tools
do not care whether the backend is Pinecone (cloud) or Chroma (local on disk):

    {"matches": [{"id": str, "score": float, "metadata": {...}}, ...]}

On failure implementations return ``{"matches": [], "error": "<message>"}`` instead
of raising, so a RAG tool degrades to "no results" rather than crashing the agent.
"""
from abc import abstractmethod
from typing import Any, Dict

from agent.db.base_db import BaseDatabase


class VectorDatabase(BaseDatabase):
    """A knowledge base that can be searched by semantic similarity."""

    @abstractmethod
    async def query(self, query: str, top_k: int = 5, include_metadata: bool = True) -> Dict[str, Any]:
        """Return the top_k records most similar to the text query."""
        raise NotImplementedError
