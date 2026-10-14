from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from typing import Any, Dict
from agent.agents.orchestrator.classify_intent import ClassifyIntent
from agent.state.agent_state import AgentState
from agent.agents.disease.disease_agent import DiseaseAgent
from agent.agents.help.help_agent import HelpAgent
from agent.agents.maintenance.maintenance_agent import MaintenanceAgent
from agent.agents.fertilize.fertilize_agent import FertilizeAgent


class OrchestratorAgent:
    def __init__(self):
        
        self.disease_agent = DiseaseAgent()
        self.fertilize_agent = FertilizeAgent()
        self.maintenance_agent = MaintenanceAgent()
        self.help_agent = HelpAgent()
        self._classify_intent = ClassifyIntent()
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph workflow: classify intent, then route to one expert agent."""
        print("Building orchestrator agent workflow")
        workflow = StateGraph(AgentState)

        # Add all nodes
        workflow.add_node("classify_intent", self._classify_intent._classify_intent)
        workflow.add_node("disease_agent", self.disease_agent.disease)
        workflow.add_node("fertilize_agent", self.fertilize_agent.fertilize)
        workflow.add_node("maintenance_agent", self.maintenance_agent.maintance)
        workflow.add_node("help_agent", self.help_agent.help)

        workflow.set_entry_point("classify_intent")

        # Define the routing logic
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
            return END  # "unknown": the classifier already set the no-answer response

        # Add conditional edges
        workflow.add_conditional_edges(
            "classify_intent",
            route_by_intent,
            {
                "disease_agent": "disease_agent",
                "fertilize_agent": "fertilize_agent",
                "maintenance_agent": "maintenance_agent",
                "help_agent": "help_agent",
                END: END,
            },
        )

        # Add edges from all agents to END
        workflow.add_edge("disease_agent", END)
        workflow.add_edge("fertilize_agent", END)
        workflow.add_edge("maintenance_agent", END)
        workflow.add_edge("help_agent", END)

        return workflow.compile()

    async def process_chat(self, message: str, userId: int, image_path: str = None,
                            callbacks=None) -> Dict[str, Any]:
        """Run the graph for a user message and return {"response": ..., "intent": ...}.

        image_path, when given, is the local path of a leaf photo the user uploaded -
        it always routes to the DiseaseAgent regardless of what the text says.
        """
        try:
            print("Processing chat request")
            print(message)
            initial_state: AgentState = {
                "userId":userId,
                "query": message,
                "intent": "",
                "response": "",
                "history": [],
                "image_path": image_path,
            }

            result = await self.graph.ainvoke(initial_state, config={"callbacks": callbacks or []})

            response_data = {
                "response": result.get("response", ""),
                "intent": result.get("intent", ""),
            }

            return response_data
        except Exception as e:
            print(f"Error processing chat: {str(e)}")
            return {
                "response": f"Error processing your request: {message}",
                "intent": "error",
            }
