"""LangGraph orchestrator for the multilingual + voice variant.

Same router as the base `agent` variant (classify the intent, send the question to one
expert), wrapped in two translation steps so the farmer can ask in Sinhala, Tamil or English:

    detect_language -> classify_intent -> <one expert | nothing> -> translate_response -> END

  detect_language     works out the language (from the script used, or the UI's choice) and
                      translates the question into English - everything downstream is English
  translate_response  translates the expert's English answer back into the farmer's language

Both steps are free no-ops for English questions (no extra LLM calls). Voice is handled
outside the graph: the backend turns speech into text before calling process_chat() and
text back into speech after.
"""
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from agent.agents.disease.disease_agent import DiseaseAgent
from agent.agents.fertilize.fertilize_agent import FertilizeAgent
from agent.agents.help.help_agent import HelpAgent
from agent.agents.maintenance.maintenance_agent import MaintenanceAgent
from agent.agents.orchestrator.classify_intent import ClassifyIntent
from agent.services.language_service import AUTO, LanguageService, normalize_language, resolve_language
from agent.state.agent_state import AgentState


class OrchestratorAgent:
    def __init__(self):
        self.disease_agent = DiseaseAgent()
        self.fertilize_agent = FertilizeAgent()
        self.maintenance_agent = MaintenanceAgent()
        self.help_agent = HelpAgent()
        self._classify_intent = ClassifyIntent()
        self.language_service = LanguageService()
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ translation nodes
    async def _detect_language(self, state: AgentState) -> Dict[str, Any]:
        """First node: pick the language and hand the experts an English question."""
        query = state["query"]
        requested = normalize_language(state.get("language")) or AUTO
        language = resolve_language(requested, query)
        english = await self.language_service.to_english(query, language)
        print(f"language: requested={requested!r} using={language!r}"
              + (f" english={english!r}" if english != query else ""))
        return {
            "language": requested,
            "detected_language": language,
            "original_query": query,
            "query": english,
        }

    async def _translate_response(self, state: AgentState) -> Dict[str, Any]:
        """Last node: put the English answer into the farmer's language."""
        language = state.get("detected_language") or "en"
        response = state.get("response", "")
        if language == "en" or not response:
            return {}
        return {"response": await self.language_service.from_english(response, language)}

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        print("Building orchestrator agent workflow (multilingual)")
        workflow = StateGraph(AgentState)

        workflow.add_node("detect_language", self._detect_language)
        workflow.add_node("classify_intent", self._classify_intent._classify_intent)
        workflow.add_node("disease_agent", self.disease_agent.disease)
        workflow.add_node("fertilize_agent", self.fertilize_agent.fertilize)
        workflow.add_node("maintenance_agent", self.maintenance_agent.maintance)
        workflow.add_node("help_agent", self.help_agent.help)
        workflow.add_node("translate_response", self._translate_response)

        workflow.set_entry_point("detect_language")
        workflow.add_edge("detect_language", "classify_intent")

        def route_by_intent(state):
            # An attached photo is always a disease/pest question, whatever the text says.
            if state.get("image_path"):
                return "disease_agent"
            intent = state.get("intent", "").lower()
            if intent == "disease":
                return "disease_agent"
            elif intent == "fertilize":
                return "fertilize_agent"
            elif intent in ("company_help", "help"):
                return "help_agent"
            # "unknown": the classifier already set the (English) no-answer message - just translate it.
            return "translate_response"

        workflow.add_conditional_edges(
            "classify_intent",
            route_by_intent,
            {
                "disease_agent": "disease_agent",
                "fertilize_agent": "fertilize_agent",
                "maintenance_agent": "maintenance_agent",
                "help_agent": "help_agent",
                "translate_response": "translate_response",
            },
        )

        # Every expert's English answer goes through the translator before leaving the graph.
        for expert in ("disease_agent", "fertilize_agent", "maintenance_agent", "help_agent"):
            workflow.add_edge(expert, "translate_response")
        workflow.add_edge("translate_response", END)

        return workflow.compile()

    async def process_chat(self, message: str, userId: int, image_path: str = None,
                           callbacks=None, language: str = AUTO) -> Dict[str, Any]:
        """Run the graph for a user message.

        Returns {"response", "intent", "language", "original_query", "translated_query"}:
          language          the language the answer is in ("en", "si", "ta")
          original_query    the message as the user wrote it
          translated_query  the English version the experts saw (None when it was already English)

        image_path, when given, is the local path of a leaf photo the user uploaded - it always
        routes to the DiseaseAgent. `language` is "auto" (detect from the text) or a code.
        """
        try:
            print(f"Processing chat request (language={language!r})")
            print(message)
            initial_state: AgentState = {
                "userId": userId,
                "query": message,
                "intent": "",
                "response": "",
                "history": [],
                "image_path": image_path,
                "language": language or AUTO,
                "detected_language": "",
                "original_query": message,
            }

            result = await self.graph.ainvoke(initial_state, config={"callbacks": callbacks or []})

            english_query = result.get("query", message)
            return {
                "response": result.get("response", ""),
                "intent": result.get("intent", ""),
                "language": result.get("detected_language") or "en",
                "original_query": result.get("original_query", message),
                "translated_query": english_query if english_query != message else None,
            }
        except Exception as e:
            print(f"Error processing chat: {str(e)}")
            return {
                "response": f"Error processing your request: {message}",
                "intent": "error",
                "language": "en",
                "original_query": message,
                "translated_query": None,
            }
