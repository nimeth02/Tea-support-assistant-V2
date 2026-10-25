"""Disease expert - built with CrewAI.

CrewAI thinks in "crews": agents with a role, goal and backstory that work on tasks. Here the
crew is small: one agent (the tea disease expert) and one task (answer this question). CrewAI
runs the reasoning loop: the agent picks tools (knowledge base, leaf-photo model, web search),
reads the results, and writes the final answer.

CrewAI has no built-in LLM fallback, so answer() tries each model from crewai_llms() in turn and
counts every switch in the a2a_expert_llm_fallbacks_total metric.
"""
import logging
from typing import Optional

from crewai import Agent, Crew, Process, Task

from agent.agents.base import ExpertAnswer
from agent.agents.disease.tools import get_disease_tools
from agent.config import settings
from agent.db.vector import get_vector_db
from agent.ops.telemetry import EXPERT_LLM_FALLBACKS
from agent.prompt_templates import disease_agent_prompt
from agent.services.llm_service import crewai_llms
from agent.state.agent_state import DiseaseTurnState

logger = logging.getLogger("agent.disease")

FALLBACK_TEXT = (
    "Sorry, I couldn't diagnose this right now. Please try again in a few minutes, or describe "
    "the leaf problem in more detail (colour, spots, insects)."
)

TASK_TEMPLATE = """Answer this tea farmer's question about a tea plant disease or pest.

<question>
{question}
</question>
{photo_note}"""
PHOTO_NOTE = "\nThe farmer attached a photo of a tea leaf. Use ClassifyLeafImage first.\n"


class DiseaseAgent:
    name = "disease"
    framework = "crewai"

    def __init__(self):
        self.vector_db = get_vector_db("disease")
        self.llms = crewai_llms()

    def build_crew(self, llm, turn: DiseaseTurnState) -> Crew:
        expert = Agent(
            role="Tea disease and pest expert",
            goal=(
                "Diagnose tea plant diseases and pests and give treatment and prevention advice "
                "that the tool results support."
            ),
            backstory=disease_agent_prompt,
            tools=get_disease_tools(self.vector_db, turn),
            llm=llm,
            allow_delegation=False,
            max_iter=6,  # stop after 6 reasoning steps so a confused model can't loop forever
            verbose=False,
        )
        task = Task(
            description=TASK_TEMPLATE.format(
                question=turn.query, photo_note=PHOTO_NOTE if turn.image_path else ""
            ),
            expected_output=(
                "A short, simple answer for a farmer: what the problem is, why it happens, how to "
                "treat it and how to prevent it. Plain text, no tool names or file paths."
            ),
            agent=expert,
        )
        return Crew(
            agents=[expert], tasks=[task], process=Process.sequential,
            max_rpm=settings.GEMINI_RPM if settings.LLM_PROVIDER.lower() == "gemini" else None,
            verbose=False,
        )

    async def answer(self, question: str, user_id: int, image_path: Optional[str] = None) -> ExpertAnswer:
        for i, llm in enumerate(self.llms):
            turn = DiseaseTurnState(user_id=user_id, query=question, image_path=image_path)
            try:
                result = await self.build_crew(llm, turn).kickoff_async()
                if i > 0:
                    EXPERT_LLM_FALLBACKS.labels(self.name, llm.model).inc()
                return ExpertAnswer(text=result.raw, tools_used=turn.tools_used, model=llm.model)
            except Exception as e:
                logger.warning("crew_failed model=%s error=%s", llm.model, e)
        return ExpertAnswer(text=FALLBACK_TEXT, degraded=True)
