"""Tools available to the disease expert: its own RAG tool, the image classifier, and the shared web search."""
from typing import List

from agent.agents.disease.tools.leaf_image_classifier import ClassifyLeafImageTool
from agent.agents.disease.tools.tea_disease_rag import TeaDiseaseRAGTool
from agent.agents.shared_tools import TeaTool, WebSearchTool
from agent.db.vector import VectorDatabase
from agent.state.agent_state import TeaChatState

EXPERT = "disease"  # intent label recorded when any of these tools runs


def get_disease_tools(vector_db: VectorDatabase, chat_state: TeaChatState) -> List[TeaTool]:
    """RAG tool bound to the given vector DB, the leaf-photo classifier, plus the shared web search fallback."""
    return [
        TeaDiseaseRAGTool(vector_db=vector_db, chat_state=chat_state, expert=EXPERT),
        ClassifyLeafImageTool(chat_state=chat_state, expert=EXPERT),
        WebSearchTool(chat_state=chat_state, expert=EXPERT),
    ]


__all__ = ["get_disease_tools", "TeaDiseaseRAGTool", "ClassifyLeafImageTool"]
