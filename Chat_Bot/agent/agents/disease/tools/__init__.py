"""Tools available to the DiseaseAgent: its own RAG tool, the image classifier, and the shared web search."""
from typing import List

from langchain.tools import Tool

from agent.agents.disease.tools.leaf_image_classifier import leaf_image_classifier_tool
from agent.agents.disease.tools.tea_disease_rag import make_tea_disease_rag_tool
from agent.agents.shared_tools import web_search_tool
from agent.db.vector import VectorDatabase


def get_disease_tools(vector_db: VectorDatabase) -> List[Tool]:
    """RAG tool bound to the given vector DB, the leaf-photo classifier, plus the shared web search fallback."""
    return [make_tea_disease_rag_tool(vector_db), leaf_image_classifier_tool, web_search_tool]


__all__ = ["get_disease_tools", "make_tea_disease_rag_tool", "leaf_image_classifier_tool"]
