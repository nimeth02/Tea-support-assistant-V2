from typing import Any, Dict, List

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, after_agent
from langchain.tools import BaseTool, ToolRuntime, tool
from langchain_core.messages import HumanMessage, ToolMessage

from agent.state.agent_state import ChatContext, TeaAgentState
from agent.agents.disease.disease_agent import DiseaseAgent
from agent.agents.help.help_agent import HelpAgent
from agent.agents.maintenance.maintenance_agent import MaintenanceAgent
from agent.agents.fertilize.fertilize_agent import FertilizeAgent
from agent.prompt_templates import orchestrator_system_prompt
from agent.services.llm_service import LLMService

# Which intent label each expert tool stands for. The labels feed the backend's metrics,
# so they match the old classifier's labels.
EXPERT_INTENTS = {
    "ask_disease_expert": "disease",
    "ask_fertilizer_expert": "fertilize",
    "ask_company_help": "company_help",
}


@after_agent(state_schema=TeaAgentState)
def record_intent(state: TeaAgentState, runtime) -> Dict[str, Any]:
    """When the agent finishes, store which experts it used in the state's `intent` field.

    e.g. "disease", "disease+fertilize" (unique, in call order), or "unknown" if it used none.
    """
    used = [
        EXPERT_INTENTS[m.name]
        for m in state["messages"]
        if isinstance(m, ToolMessage) and m.name in EXPERT_INTENTS
    ]
    return {"intent": "+".join(dict.fromkeys(used)) or "unknown"}


class OrchestratorAgent:
    """Supervisor agent: the LLM decides which expert agents to call, then writes the reply.

    Each expert agent is wrapped as a tool. The LLM runs in a loop - call an expert, read its
    answer, call another or finish - so a mixed question ("spots on my leaves, and how much
    fertilizer?") can be sent to several experts and answered in one reply.
    """

    def __init__(self):

        self.disease_agent = DiseaseAgent()
        self.fertilize_agent = FertilizeAgent()
        self.maintenance_agent = MaintenanceAgent()  # not offered as a tool yet - it has no LLM wired in
        self.help_agent = HelpAgent()
        self.llm = LLMService().get_llm()
        self.agent = create_agent(
            self.llm,
            tools=self._expert_tools(),
            system_prompt=orchestrator_system_prompt,
            state_schema=TeaAgentState,
            context_schema=ChatContext,
            middleware=[
                # Raise an error after 6 LLM calls so a confused model can't loop forever.
                ModelCallLimitMiddleware(run_limit=6, exit_behavior="error"),
                record_intent,
            ],
            name="orchestrator",
        )

    def _expert_tools(self) -> List[BaseTool]:
        """Wrap each expert agent as a tool the supervisor can call.

        The LLM only writes the `question`. `runtime` is filled in by LangChain and hidden from
        the LLM: through it each tool reads this turn's ChatContext (userId, photo path).
        """
        disease, fertilize, company_help = self.disease_agent, self.fertilize_agent, self.help_agent

        @tool(parse_docstring=True)
        async def ask_disease_expert(question: str, runtime: ToolRuntime[ChatContext]) -> str:
            """Tea plant diseases and pests: spots or discoloured leaves, insects, fungus, wilting,
            dieback, damaged stems or roots, and how to treat or prevent them. Also identifies the
            disease in an attached leaf photo.

            Args:
                question: The farmer's question for this expert, rewritten so it makes sense on its own.
            """
            return await disease.answer(question, runtime.context)

        @tool(parse_docstring=True)
        async def ask_fertilizer_expert(question: str, runtime: ToolRuntime[ChatContext]) -> str:
            """Fertilizer type and dosage, when to fertilize, the farmer's fertilization history,
            rain or weather forecast for the farmer's land, and details of the farmer's tea land
            (location, size, slope, number of plants).

            Args:
                question: The farmer's question for this expert, rewritten so it makes sense on its own.
            """
            return await fertilize.answer(question, runtime.context)

        @tool(parse_docstring=True)
        async def ask_company_help(question: str, runtime: ToolRuntime[ChatContext]) -> str:
            """The tea company and its services: green leaf collection days and times, leaf quality
            and deductions, payments, advances, loans, supplier registration, complaints, policies.

            Args:
                question: The farmer's question for this expert, rewritten so it makes sense on its own.
            """
            return await company_help.answer(question, runtime.context)

        return [ask_disease_expert, ask_fertilizer_expert, ask_company_help]

    async def process_chat(self, message: str, userId: int, image_path: str = None,
                            callbacks=None) -> Dict[str, Any]:
        """Run the supervisor for a user message and return {"response": ..., "intent": ...}.

        image_path, when given, is the local path of a leaf photo the user uploaded. The
        supervisor is only told a photo is attached; the disease expert gets the path itself
        from the ChatContext.
        """
        try:
            print("Processing chat request")
            print(message)
            context = ChatContext(user_id=userId, image_path=image_path)

            text = message
            if context.image_path:
                text += "\n\n(The farmer attached a photo of a tea leaf.)"

            result = await self.agent.ainvoke(
                {"messages": [HumanMessage(text)]},
                context=context,
                config={"callbacks": callbacks or []},
            )

            return {
                "response": result["messages"][-1].text,
                "intent": result.get("intent", "unknown"),
            }
        except Exception as e:
            print(f"Error processing chat: {str(e)}")
            return {
                "response": f"Error processing your request: {message}",
                "intent": "error",
            }
