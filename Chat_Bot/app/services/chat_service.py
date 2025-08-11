from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from app.models.chat import ChatRequest, ChatResponse
from app.services.classify_intent_service import ClassifyIntent
from app.state.chat_state import ChatState
from app.services.disease_service import DiseaseService
from app.services.help_service import HelpService
from app.services.maintainece_service import MaintanceService
from app.services.fertilize_service import FertilizeService


class ChatService:
    def __init__(self):
        
        self.disease_service = DiseaseService()
        self.fertilize_service = FertilizeService()
        self.maintainece_service = MaintanceService()
        self.help_service = HelpService()
        self._classify_intent = ClassifyIntent()
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph workflow with complete routing for all services."""
        print("Building chat service workflow")
        workflow = StateGraph(ChatState)

        # Add all nodes
        workflow.add_node("classify_intent", self._classify_intent._classify_intent)
        workflow.add_node("disease_service", self.disease_service.disease)
        workflow.add_node("fertilize_service", self.fertilize_service.fertilize)
        workflow.add_node("maintenance_service", self.maintainece_service.maintance)
        workflow.add_node("help_service", self.help_service.help)

        workflow.set_entry_point("classify_intent")

        # Define the routing logic
        def route_by_intent(state):
            intent = state.get("intent", "").lower()
            if intent == "disease":
                return "disease_service"
            elif intent == "fertilize":
                return "fertilize_service"
            elif intent == "help":
                return "help_service"
            return END

        # Add conditional edges
        workflow.add_conditional_edges(
            "classify_intent",
            route_by_intent,
            {
                "disease_service": "disease_service",
                "fertilize_service": "fertilize_service",
                "maintenance_service": "maintenance_service",
                "help_service": "help_service",
                END: END,
            },
        )

        # Add edges from all services to END
        workflow.add_edge("disease_service", END)
        workflow.add_edge("fertilize_service", END)
        workflow.add_edge("maintenance_service", END)
        workflow.add_edge("help_service", END)

        return workflow.compile()

    async def process_chat(self, chat_request: ChatRequest,userId:int) -> ChatResponse:
        # First try to answer using FAQ service
        try:
            print("Processing chat request")
            print(chat_request)
            initial_state: ChatState = {
                "userId":userId,
                "query": chat_request.message,
                "intent": "",
                "response": "",
                "history": [],
            }

            result = await self.graph.ainvoke(initial_state)

            response_data = {
                "response": result.get("response", ""),
                "intent": result.get("intent", ""),
            }

            return response_data
        except Exception as e:
            print(f"Error processing chat: {str(e)}")
            return ChatResponse(
                response=f"Error processing your request: {chat_request.message}",
                intent="error",
            )
