from langchain_core.prompts import ChatPromptTemplate
from app.services.llm_service import LLMService
from typing import Dict, Any
import httpx


class ClassifyIntent:
    def __init__(self):
        self.llm = LLMService().get_llm()

    async def _classify_intent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Classify the query's intent using Groq LLM."""
        try:

            if "query" not in state or not isinstance(state["query"], str):
                raise ValueError("Invalid or missing query in state")

            prompt = ChatPromptTemplate.from_template(
                """You are an expert in tea farming support. 
                    Classify the user's intent from the following categories:
                    - fertilize
                    - disease
                    - company_help
                    - unknown

                    explanation
                    - fertilize = fertilize history, whether conditions, land details
                    - disease = diseases, symptons
                    - company_help = tea collecting, advance , loans, comapny policies
                    - unknown = other unrelated tuffs

                    Here are examples:
                    Query: "My tea plants have yellow leaves." → disease  
                    Query: "How much urea should I apply?" → fertilize 
                    Query: "Where is my tea land?" → fertilize 
                    Query: "How much rain tomarrow?" → fertilize 
                    Query: "which date company not collect tea leaves?" → company_help  
                    Query: "Tell me about AI." → unknown
                    Query: "which Car is good BYD or Toyota" → unknown

                    Now classify this query:
                    Query: {query}

                    Respond with only one word: fertilize, disease, company_help, or unknown."""
            )

            try:
                response = await self.llm.ainvoke(prompt.format(query=state["query"]))
            except httpx.ConnectError as ce:
                print(f"Connection error with Groq API: {str(ce)}")
                raise RuntimeError("Could not connect to AI service")
            except Exception as e:
                print(f"Groq API error: {str(e)}")
                raise RuntimeError("Error processing with AI service")
            
            intent = response.content.strip().lower()
            print(f"classify intent : {intent}")
            if intent == "unknown":
                {
                    "intent": "unknown",
                    "history": [],
                    "query": state["query"],
                    "response": "I'm not sure how to respond to that. Could you please rephrase?",
                }

            return {
                "intent": intent,
                "history": [],
                "query": state["query"],
                "response": "",
            }
        except (ValueError, KeyError, RuntimeError) as e:
            print(f"Error classifying intent: {str(e)}")
            return {
                "intent": "unknown",
                "history": [f"Intent classification failed: {str(e)}"],
                "query": state.get("query", ""),
                "response": "",
            }
