from app.db.help_vector_db import PineconeDatabase
from langchain_core.prompts import ChatPromptTemplate
from app.services.llm_service import LLMService
from typing import Dict, Any, List


class HelpService:
    def __init__(self):
        self.vector_db = PineconeDatabase()
        self.llm = LLMService().get_llm()

    async def help(self, state: Dict[str, Any]) -> Dict[str, Any]:

        retrieved_records = await self._retrieve_from_pinecone(state["query"])

        # Step 2: Generate response using retrieved context
        response_text = await self._generate_response(state["query"], retrieved_records)

        return {
            "intent": state["intent"],
            "history": state["history"],
            "query": state["query"],
            "response": response_text or "There is no help",
        }

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

        prompt = ChatPromptTemplate.from_template(
            "User Query: {query}\n\n"
            "Help Database (context):\n{context}\n\n"
            "Instructions:\n"
            "• Answer only if the query clearly matches the help topics.\n"
            "• Help topics include: company policies, tea leaf collection process, salary, loans/advances, attendance, and other support.\n"
            "• Be short, polite, and easy to understand — like you're helping a colleague.\n"
            "• If the query doesn’t match any context, say: 'Sorry, I couldn’t find help related to that question.'\n\n"
            "Respond in a short and helpful tone."
        )

        try:
            response = await self.llm.ainvoke(
                prompt.format(query=query, context=context)
            )
            print(response)
            return response.content.strip().lower()
        except Exception as e:
            return f"Failed to generate response: {str(e)}"
