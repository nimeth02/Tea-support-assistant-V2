"""Company help expert - built with LangGraph.

LangGraph describes the work as a graph of steps ("nodes") joined by arrows ("edges"). Here the
path is fixed and easy to follow, which suits a help desk that must only answer from its
database:

    START -> retrieve -> (records found?) --yes--> answer   -> END
                                           --no---> decline  -> END

  retrieve  searches the company-help vector DB (no LLM call)
  answer    one LLM call that may only use the retrieved records
  decline   a fixed "couldn't find help" reply, with no LLM call at all (cheaper and safer
            than asking the model to refuse)
"""
import logging
from typing import Optional

from langgraph.graph import END, START, StateGraph

from agent.agents.base import ExpertAnswer
from agent.db.vector import get_vector_db
from agent.prompt_templates import help_prompt
from agent.services.llm_service import langchain_llm
from agent.state.agent_state import HelpState

logger = logging.getLogger("agent.help")

NO_HELP_TEXT = "Sorry, I couldn't find help related to that question."
FALLBACK_TEXT = "Sorry, the help desk can't answer right now. Please try again in a few minutes."
MIN_RELEVANCE = 0.2  # records scoring below this are treated as "not found"


class HelpAgent:
    name = "help"
    framework = "langgraph"

    def __init__(self):
        self.vector_db = get_vector_db("help")
        self.llm = langchain_llm()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(HelpState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("answer", self._answer)
        graph.add_node("decline", self._decline)
        graph.add_edge(START, "retrieve")
        graph.add_conditional_edges(
            "retrieve", lambda s: "answer" if s.get("records") else "decline",
            {"answer": "answer", "decline": "decline"},
        )
        graph.add_edge("answer", END)
        graph.add_edge("decline", END)
        return graph.compile(name="help_agent")

    async def _retrieve(self, state: HelpState) -> HelpState:
        results = await self.vector_db.query(query=state["question"], top_k=3, include_metadata=True)
        records = [
            {"question": m["metadata"].get("question", ""), "answer": m["metadata"].get("answer", "")}
            for m in results.get("matches", [])
            if m.get("score") is None or m["score"] >= MIN_RELEVANCE
        ]
        return {"records": records}

    async def _answer(self, state: HelpState) -> HelpState:
        context = "\n\n".join(f"- question: {r['question']}\n  answer: {r['answer']}" for r in state["records"])
        response = await self.llm.ainvoke(help_prompt.format_messages(query=state["question"], context=context))
        return {"answer": response.text.strip()}

    async def _decline(self, state: HelpState) -> HelpState:
        return {"answer": NO_HELP_TEXT}

    async def answer(self, question: str, user_id: int, image_path: Optional[str] = None) -> ExpertAnswer:
        try:
            result = await self.graph.ainvoke({"question": question})
        except Exception as e:
            logger.warning("help_graph_failed error=%s", e)
            return ExpertAnswer(text=FALLBACK_TEXT, degraded=True)
        tools = ["CompanyHelpRAG"] + (["HelpAnswerLLM"] if result.get("records") else [])
        return ExpertAnswer(text=result["answer"] or NO_HELP_TEXT, tools_used=tools)
