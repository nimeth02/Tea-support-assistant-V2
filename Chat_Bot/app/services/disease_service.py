from app.db.disease_vector_db import PineconeDatabase
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from app.services.llm_service import LLMService
from typing import Dict, Any, List
from duckduckgo_search import DDGS
import asyncio


class DiseaseService:
    def __init__(self):
        self.vector_db = PineconeDatabase()
        self.llm = LLMService().get_llm()
        self.tools = self._initialize_tools()
        self.agent = self._initialize_agent()
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def _initialize_tools(self):
        return [
            Tool(
                name="TeaDiseaseRAG",
                func=lambda query: asyncio.run(self._rag_search(query)),
                description="Retrieves disease or pest information from a curated tea agriculture database. Use for detailed, accurate, and local information.",
            ),
            Tool(
                name="DuckDuckGoWebSearch",
                func=lambda query: self.duckduckgo_search(query),
                description="Searches the web for general disease or pest info in tea, if local knowledge base is insufficient.",
            )
        ]

    def _initialize_agent(self):
        template = """You are an expert assistant in diagnosing tea plant diseases and pests.
                You have access to the following tools:

                {tools}

                Use the following format:

                Question: the input question you must answer
                Thought: you should always think about what to do
                Action: the action to take, should be one of [{tool_names}]
                Action Input: the input to the action
                Observation: the result of the action
                ... (this Thought/Action/Action Input/Observation can repeat N times)
                Thought: I now know the final answer
                Final Answer: the final answer to the original input question

                Guidelines:
                - Final answer should be very simple understandable short and structured for low knowleded farmer
                - Guide farmer for solutions he can apply simply than technical words
                - why this happen and how it avoid

                Question: {input}
                UserId: {userId}
                Thought:{agent_scratchpad}"""

        prompt = ChatPromptTemplate.from_template(template).partial(
            tool_names=", ".join([tool.name for tool in self.tools])
        )
        return create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)

    async def disease(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            input_data = {
                "input": state["query"],
                "userId": state.get("user_id", "unknown"),
            }

            result = await self.agent_executor.ainvoke(input_data)

            return {
                "intent": state["intent"],
                "history": state["history"],
                "query": state["query"],
                "response": result.get("output", "Sorry, I couldn’t determine the disease."),
            }
        except Exception as e:
            return {
                "intent": state["intent"],
                "history": state["history"],
                "query": state["query"],
                "response": f"Agent execution failed: {str(e)}",
            }

    async def _rag_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant disease/pest records from Pinecone based on the query.
        Assumes text embeddings are stored in Pinecone.
        """
        try:
            # Query Pinecone for top-k relevant records (e.g., top 5)
            results = await self.vector_db.query(
                query=query, top_k=3, include_metadata=True
            )
            print(results)
            # Extract records from Pinecone response
            matches = results.get("matches", [])
            records = [
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
            return records
        except Exception as e:
            return [{"error": f"Failed to retrieve from Pinecone: {str(e)}"}]

    def duckduckgo_search(self,query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = ddgs.text(
                    query, region="in-en", safesearch="moderate", max_results=3
                )
                print(results)
                if not results:
                    return "No results found from the web."
                return "\n\n".join([f"- {r['title']}: {r['href']}" for r in results])
        except Exception as e:
            return f"DuckDuckGo search failed: {str(e)}"


# from app.db.disease_vector_db import PineconeDatabase
# from langchain_core.prompts import ChatPromptTemplate
# from app.services.llm_service import LLMService
# from typing import Dict, Any, List


# class DiseaseService:
#     def __init__(self):
#         self.vector_db = PineconeDatabase()
#         self.llm = LLMService().get_llm()


#     async def disease(self, state: Dict[str, Any]) -> Dict[str, Any]:

#         retrieved_records = await self._retrieve_from_pinecone(state["query"])

#         response_text = await self._generate_response(state["query"], retrieved_records)

#         return {
#             "intent": state["intent"],
#             "history": state["history"],
#             "query": state["query"],
#             "response": response_text or "This is critical disease",
#         }

#     async def _retrieve_from_pinecone(self, query: str) -> List[Dict[str, Any]]:
#         """
#         Retrieve relevant disease/pest records from Pinecone based on the query.
#         Assumes text embeddings are stored in Pinecone.
#         """
#         try:
#             # Query Pinecone for top-k relevant records (e.g., top 5)
#             results = await self.vector_db.query(
#                 query=query, top_k=3, include_metadata=True
#             )
#             # Extract records from Pinecone response
#             matches = results.get("matches", [])
#             records = [
#                 {
#                     "id": match["id"],
#                     "text": match["metadata"].get("text", ""),
#                     "category": match["metadata"].get("category", ""),
#                     "disease_or_pest": match["metadata"].get(
#                         "disease", match["metadata"].get("pest", "")
#                     ),
#                     "instruction": match["metadata"].get("instruction", ""),
#                 }
#                 for match in matches
#             ]
#             return records
#         except Exception as e:
#             return [{"error": f"Failed to retrieve from Pinecone: {str(e)}"}]

#     async def _generate_response(
#         self, query: str, records: List[Dict[str, Any]]
#     ) -> str:
#         if not records or "error" in records[0]:
#             return "Sorry, I couldn't retrieve relevant disease information. Please try again."

#         # Combine all the record texts as context
#         context = "\n\n".join(
#             [
#                 f"- {r['disease_or_pest']} ({r['category']}): {r['text']}\nInstruction: {r['instruction']}"
#                 for r in records
#             ]
#         )

#         prompt = ChatPromptTemplate.from_template(
#             "You are a tea plant disease and pest expert assistant.\n\n"
#             "User Query:\n{query}\n\n"
#             "Relevant Information from Knowledge Base:\n{context}\n\n"
#             "Instructions:\n"
#             "- Answer clearly and concisely in 2–4 sentences.\n"
#             "- Use bullet points if listing symptoms, instructions, or treatments.\n"
#             "- If a specific disease or pest is found, include:\n"
#             "  • Recommend treatments or actions that are realistic in rural or developing areas of Sri Lanka.\n"
#             "  • Include local or traditional practices if appropriate.\n"
#             "  • Use bullet points or short steps to keep it easy to follow.\n"
#             "  • If the context is not relevant, say so politely and give general advice for tea disease management.\n\n"
#             "  • Key symptoms\n"
#             "Now respond to the query using the above guidelines."
#         )

#         try:
#             response = await self.llm.ainvoke(
#                 prompt.format(query=query, context=context)
#             )
#             return response.content.strip().lower()
#         except Exception as e:
#             return f"Failed to generate response: {str(e)}"
