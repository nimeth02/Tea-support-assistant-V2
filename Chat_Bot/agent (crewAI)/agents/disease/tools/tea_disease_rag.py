"""TeaDiseaseRAG tool: retrieves disease / pest records from the tea-disease knowledge base."""
import asyncio
from typing import Any, Dict, List, Type

from pydantic import BaseModel, Field

from agent.agents.shared_tools.base import TeaTool
from agent.agents.shared_tools.schemas import SearchQuery
from agent.db.vector import VectorDatabase


async def rag_search(vector_db: VectorDatabase, query: str) -> List[Dict[str, Any]]:
    """Retrieve the top matching disease/pest records for the query."""
    try:
        results = await vector_db.query(query=query, top_k=3, include_metadata=True)
        print(results)
        matches = results.get("matches", [])
        return [
            {
                "id": match["id"],
                "text": match["metadata"].get("text", ""),
                "category": match["metadata"].get("category", ""),
                "disease_or_pest": match["metadata"].get(
                    "disease", match["metadata"].get("pest", "")
                ),
                "instruction": match["metadata"].get("instruction", ""),
            }
            for match in matches
        ]
    except Exception as e:
        return [{"error": f"Failed to retrieve from Pinecone: {str(e)}"}]


class TeaDiseaseRAGTool(TeaTool):
    name: str = "TeaDiseaseRAG"
    description: str = (
        "Retrieves disease or pest information from a curated tea agriculture database. "
        "Use for detailed, accurate, and local information."
    )
    args_schema: Type[BaseModel] = SearchQuery
    vector_db: Any = Field(exclude=True)  # Chroma or Pinecone client, set when the tool is built

    def _run(self, query: str) -> List[Dict[str, Any]]:
        self.note_use()
        # CrewAI runs tools in a worker thread with no event loop, so asyncio.run is safe here.
        return asyncio.run(rag_search(self.vector_db, query))
