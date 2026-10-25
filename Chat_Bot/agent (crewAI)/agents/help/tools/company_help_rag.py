"""CompanyHelpRAG tool: retrieves question/answer records from the company-help knowledge base."""
import asyncio
from typing import Any, Dict, List, Type

from pydantic import BaseModel, Field

from agent.agents.shared_tools.base import TeaTool
from agent.agents.shared_tools.schemas import SearchQuery
from agent.db.vector import VectorDatabase


async def help_search(vector_db: VectorDatabase, query: str) -> List[Dict[str, Any]]:
    """Retrieve the top matching help records (question + answer) for the query."""
    try:
        results = await vector_db.query(query=query, top_k=3, include_metadata=True)
        records = [
            {
                "id": match["id"],
                "question": match["metadata"].get("question", ""),
                "answer": match["metadata"].get("answer", ""),
            }
            for match in results.get("matches", [])
        ]
        print(records)
        return records or [{"error": "No matching help records found."}]
    except Exception as e:
        return [{"error": f"Failed to retrieve help records: {str(e)}"}]


class CompanyHelpRAGTool(TeaTool):
    name: str = "CompanyHelpRAG"
    description: str = (
        "Searches the tea company's help database (leaf collection, leaf quality, payments, "
        "advances, loans, registration, complaints, policies) and returns matching questions with "
        "their official answers."
    )
    args_schema: Type[BaseModel] = SearchQuery
    vector_db: Any = Field(exclude=True)  # Chroma or Pinecone client, set when the tool is built

    def _run(self, query: str) -> List[Dict[str, Any]]:
        self.note_use()
        # CrewAI runs tools in a worker thread with no event loop, so asyncio.run is safe here.
        return asyncio.run(help_search(self.vector_db, query))
