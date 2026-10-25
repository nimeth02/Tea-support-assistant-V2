from typing import Any, Dict, List

from crewai import LLM, Agent, Crew, Process, Task
from crewai.flow.flow import Flow, start

from agent.agents.disease.disease_agent import DiseaseAgent
from agent.agents.fertilize.fertilize_agent import FertilizeAgent
from agent.agents.help.help_agent import HelpAgent
from agent.agents.maintenance.maintenance_agent import MaintenanceAgent
from agent.config import settings
from agent.prompt_templates import orchestrator_prompt as p
from agent.services.llm_service import LLMService
from agent.state.agent_state import TeaChatState


class TeaChatFlow(Flow[TeaChatState]):
    """CrewAI Flow for one chat turn. Its state (TeaChatState) is the state schema.

    The flow has a single step: run the hierarchical crew. The crew's manager agent decides on
    its own which expert agents to delegate to - one, several, or none for an off-topic message.
    """

    def __init__(self, orchestrator: "OrchestratorAgent", **kwargs: Any):
        # suppress_flow_events: skip CrewAI's emoji console panels for every flow run - they are
        # just noise in the server log, and crash with "'charmap' codec" errors in a Windows console.
        super().__init__(suppress_flow_events=True, **kwargs)
        self.orchestrator = orchestrator

    @start()
    async def run_crew(self) -> str:
        last_error: Exception | None = None

        # Try the primary model, then each fallback, until one run succeeds.
        for llm in self.orchestrator.llms:
            self.state.experts_used.clear()
            try:
                result = await self.orchestrator.build_crew(llm, self.state).kickoff_async()
                self.state.response = result.raw
                break
            except Exception as e:
                print(f"Crew run failed with {llm.model}: {e}")
                last_error = e
        else:
            raise last_error or RuntimeError("No LLM configured")

        # Which experts' tools ran, e.g. "disease+fertilize"; none means the manager answered alone.
        self.state.intent = "+".join(self.state.experts_used) or "unknown"
        return self.state.response


class OrchestratorAgent:
    """Supervisor: a CrewAI hierarchical crew whose manager delegates to the expert agents."""

    def __init__(self):

        self.disease_agent = DiseaseAgent()
        self.fertilize_agent = FertilizeAgent()
        self.maintenance_agent = MaintenanceAgent()  # not added to the crew yet - placeholder tools only
        self.help_agent = HelpAgent()
        self.llms: List[LLM] = LLMService().get_llms()

    def build_crew(self, llm: LLM, state: TeaChatState) -> Crew:
        """Build the crew for one chat turn: the manager, the expert agents and the one task."""
        experts = [
            self.disease_agent.build(llm, state),
            self.fertilize_agent.build(llm, state),
            self.help_agent.build(llm, state),
        ]
        manager = Agent(
            role=p.MANAGER_ROLE,
            goal=p.MANAGER_GOAL,
            backstory=p.MANAGER_BACKSTORY,
            llm=llm,
            allow_delegation=True,  # gives it the "Delegate work to coworker" tools
            max_iter=8,  # stop after 8 reasoning steps so a confused model can't loop forever
            verbose=False,
        )
        task = Task(
            description=p.TASK_TEMPLATE.format(
                message=state.query,
                photo_note=p.PHOTO_NOTE if state.image_path else "",
            ),
            expected_output=p.EXPECTED_OUTPUT,
        )
        return Crew(
            agents=experts,
            tasks=[task],
            process=Process.hierarchical,
            manager_agent=manager,
            # Gemini free tier: CrewAI waits when this many LLM calls per minute is reached.
            max_rpm=settings.GEMINI_RPM if settings.LLM_PROVIDER.lower() == "gemini" else None,
            verbose=False,
        )

    async def process_chat(self, message: str, userId: int, image_path: str = None,
                            callbacks=None) -> Dict[str, Any]:
        """Run the flow for a user message and return {"response": ..., "intent": ...}.

        image_path, when given, is the local path of a leaf photo the user uploaded. The manager
        is only told a photo is attached; the disease expert's tool reads the path from the state.

        callbacks are LangChain callback handlers, accepted so the backend can call this the same
        way as the LangChain versions. CrewAI does not use LangChain callbacks, so they are ignored.
        """
        try:
            print("Processing chat request")
            print(message)
            flow = TeaChatFlow(self)
            await flow.kickoff_async(
                inputs={"user_id": userId, "query": message, "image_path": image_path}
            )

            return {
                "response": flow.state.response,
                "intent": flow.state.intent,
            }
        except Exception as e:
            print(f"Error processing chat: {str(e)}")
            return {
                "response": f"Error processing your request: {message}",
                "intent": "error",
            }
