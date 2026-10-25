"""TeaDiseaseRAG tool: retrieves disease / pest records from the Pinecone tea-disease index."""
import asyncio
from typing import Any, Dict, List

from langchain_core.tools import Tool

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


def make_tea_disease_rag_tool(vector_db: VectorDatabase) -> Tool:
    """Build the RAG tool bound to a vector DB (Chroma or Pinecone)."""
    return Tool(
        name="TeaDiseaseRAG",
        func=lambda query: asyncio.run(rag_search(vector_db, query)),
        args_schema=SearchQuery,
        description=(
            "Retrieves disease or pest information from a curated tea agriculture database. "
            "Use for detailed, accurate, and local information."
        ),
    )
