from agent.db.vector import get_vector_db
from agent.prompt_templates import help_prompt
from agent.services.llm_service import LLMService
from agent.state.agent_state import ChatContext
from typing import Dict, Any, List


class HelpAgent:
    """RAG answerer for tea-company questions: fetch matching Q&A records, then let the LLM answer."""

    def __init__(self):
        self.vector_db = get_vector_db("help")
        self.llm = LLMService().get_llm()

    async def answer(self, question: str, context: ChatContext) -> str:

        retrieved_records = await self._retrieve_from_pinecone(question)

        # Step 2: Generate response using retrieved context
        response_text = await self._generate_response(question, retrieved_records)

        return response_text or "There is no help"

    async def _retrieve_from_pinecone(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant disease/pest records from Pinecone based on the query.
        Assumes text embeddings are stored in Pinecone.
        """
        try:
            # Query Pinecone for top-k relevant records (e.g., top 5)
            results = await self.vector_db.query(
                query=query, top_k=3, include_metadata=True
            )

            matches = results.get("matches", [])
            records = [
                {
                    "id": match["id"],
                    "question": match["metadata"].get("question", ""),
                    "answer": match["metadata"].get("answer", ""),
                }
                for match in matches
            ]

            print(records)
            return records
        except Exception as e:
            return [{"error": f"Failed to retrieve from Pinecone: {str(e)}"}]

    async def _generate_response(
        self, query: str, records: List[Dict[str, Any]]
    ) -> str:
        if not records or "error" in records[0]:
            return "Sorry, I couldn't retrieve relevant help information. Please try again."

        # Combine all the record texts as context
        context = "\n\n".join(
            [f"- question:{r['question']}:\nanswer: {r['answer']}" for r in records]
        )

        try:
            response = await self.llm.ainvoke(
                help_prompt.format(query=query, context=context)
            )
            print(response)
            return response.text.strip()
        except Exception as e:
            return f"Failed to generate response: {str(e)}"
