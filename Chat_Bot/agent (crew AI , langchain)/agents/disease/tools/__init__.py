"""CrewAI tools of the disease expert, built fresh for each question (they share its state)."""
from typing import List

from agent.agents.disease.tools.base import TeaTool
from agent.agents.disease.tools.leaf_image_classifier import ClassifyLeafImageTool, classify_leaf_image
from agent.agents.disease.tools.tea_disease_rag import TeaDiseaseRAGTool, WebSearchTool, rag_search
from agent.db.vector import VectorDatabase
from agent.state.agent_state import DiseaseTurnState


def get_disease_tools(vector_db: VectorDatabase, turn: DiseaseTurnState) -> List[TeaTool]:
    return [
        TeaDiseaseRAGTool(vector_db=vector_db, turn=turn),
        ClassifyLeafImageTool(turn=turn),
        WebSearchTool(turn=turn),
    ]


__all__ = ["get_disease_tools", "classify_leaf_image", "rag_search", "TeaTool"]
