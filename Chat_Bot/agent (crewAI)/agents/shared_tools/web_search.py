"""DuckDuckGoWebSearch tool (shared): web fallback when a local knowledge base is insufficient."""
from typing import Type

from duckduckgo_search import DDGS
from pydantic import BaseModel

from agent.agents.shared_tools.base import TeaTool
from agent.agents.shared_tools.schemas import SearchQuery


def duckduckgo_search(query: str) -> str:
    """Return up to three web results as a bullet list of title and link."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region="in-en", safesearch="moderate", max_results=3)
            print(results)
            if not results:
                return "No results found from the web."
            return "\n\n".join([f"- {r['title']}: {r['href']}" for r in results])
    except Exception as e:
        return f"DuckDuckGo search failed: {str(e)}"


class WebSearchTool(TeaTool):
    name: str = "DuckDuckGoWebSearch"
    description: str = (
        "Searches the web for general tea cultivation information (diseases, pests, fertilizer, weather, practices) "
        "when the local knowledge base is insufficient."
    )
    args_schema: Type[BaseModel] = SearchQuery

    def _run(self, query: str) -> str:
        self.note_use()
        return duckduckgo_search(query)
