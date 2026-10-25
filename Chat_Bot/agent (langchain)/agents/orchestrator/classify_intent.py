import re
from typing import Any, Dict

import httpx

from agent.prompt_templates import classify_intent_prompt
from agent.prompt_templates.classify_intent_prompt import INTENT_LABELS
from agent.prompt_templates.orchestrator_prompt import UNKNOWN_RESPONSE
from agent.services.llm_service import LLMService


def parse_intent(raw: str) -> str:
    """Map the model's reply to one of INTENT_LABELS.

    The prompt asks for a single word, but small models sometimes add punctuation or a short
    phrase ("Intent: disease."). Take the first known label found; anything else is unknown.
    """
    text = (raw or "").strip().lower()
    for label in INTENT_LABELS:
        if re.search(rf"\b{label}\b", text):
            return label
    return "unknown"


class ClassifyIntent:
    def __init__(self):
        self.llm = LLMService().get_llm()

    async def _classify_intent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Classify the farmer's message into one intent label (first step of the orchestrator chain)."""
        try:
            if "query" not in state or not isinstance(state["query"], str):
                raise ValueError("Invalid or missing query in state")

            try:
                response = await self.llm.ainvoke(classify_intent_prompt.format(query=state["query"]))
            except httpx.ConnectError as ce:
                print(f"Connection error with LLM API: {str(ce)}")
                raise RuntimeError("Could not connect to AI service")
            except Exception as e:
                print(f"LLM API error: {str(e)}")
                raise RuntimeError("Error processing with AI service")

            intent = parse_intent(response.text)
            print(f"classify intent : {intent!r} (raw: {response.text.strip()!r})")

            return {
                "intent": intent,
                "history": [],
                "query": state["query"],
                # Unrelated / general questions get no answer; the chain ends here with this message.
                "response": UNKNOWN_RESPONSE if intent == "unknown" else "",
            }
        except (ValueError, KeyError, RuntimeError) as e:
            print(f"Error classifying intent: {str(e)}")
            return {
                "intent": "unknown",
                "history": [f"Intent classification failed: {str(e)}"],
                "query": state.get("query", ""),
                "response": "",
            }
