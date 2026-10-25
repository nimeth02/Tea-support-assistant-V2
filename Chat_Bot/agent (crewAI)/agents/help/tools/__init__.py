"""Tools available to the company help desk expert."""
from typing import List

from agent.agents.help.tools.company_help_rag import CompanyHelpRAGTool
from agent.agents.shared_tools import TeaTool
from agent.db.vector import VectorDatabase
from agent.state.agent_state import TeaChatState

EXPERT = "company_help"  # intent label recorded when any of these tools runs


def get_help_tools(vector_db: VectorDatabase, chat_state: TeaChatState) -> List[TeaTool]:
    return [CompanyHelpRAGTool(vector_db=vector_db, chat_state=chat_state, expert=EXPERT)]


__all__ = ["get_help_tools", "CompanyHelpRAGTool"]
