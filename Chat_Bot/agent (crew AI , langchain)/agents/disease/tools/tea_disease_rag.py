"""TeaDiseaseRAG + DuckDuckGoWebSearch: the disease expert's CrewAI knowledge tools."""
import asyncio
from typing import Any, Dict, List, Type

from pydantic import BaseModel, Field

from agent.agents.disease.tools.base import TeaTool
from agent.agents.shared_tools import duckduckgo_search
from agent.agents.shared_tools.schemas import SearchQuery
from agent.db.vector import VectorDatabase


async def rag_search(vector_db: VectorDatabase, query: str) -> List[Dict[str, Any]]:
    """Retrieve the top matching disease/pest records for the query."""
    try:
        results = await vector_db.query(query=query, top_k=3, include_metadata=True)
    except Exception as e:
        return [{"error": f"Knowledge base lookup failed: {e}"}]
    records = [
        {
            "id": m["id"],
            "text": m["metadata"].get("text", ""),
            "category": m["metadata"].get("category", ""),
            "disease_or_pest": m["metadata"].get("disease", m["metadata"].get("pest", "")),
            "instruction": m["metadata"].get("instruction", ""),
        }
        for m in results.get("matches", [])
    ]
    return records or [{"error": "No matching disease records found."}]


class TeaDiseaseRAGTool(TeaTool):
    name: str = "TeaDiseaseRAG"
    description: str = (
        "Retrieves disease or pest information from a curated tea agriculture database. "
        "Use for detailed, accurate, and local information."
    )
    args_schema: Type[BaseModel] = SearchQuery
    vector_db: Any = Field(exclude=True)  # Chroma or Pinecone client

    def _run(self, query: str) -> List[Dict[str, Any]]:
        self.note_use()
        # CrewAI runs tools in a worker thread with no event loop, so asyncio.run is safe here.
        return asyncio.run(rag_search(self.vector_db, query))


class WebSearchTool(TeaTool):
    name: str = "DuckDuckGoWebSearch"
    description: str = (
        "Searches the web for general tea disease and pest information when the local "
        "knowledge base has no answer."
    )
    args_schema: Type[BaseModel] = SearchQuery

    def _run(self, query: str) -> str:
        self.note_use()
        return duckduckgo_search(query)
