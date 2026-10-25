"""DuckDuckGoWebSearch tool (shared): web fallback when a local knowledge base is insufficient."""
from duckduckgo_search import DDGS
from langchain_core.tools import Tool

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


web_search_tool = Tool(
    name="DuckDuckGoWebSearch",
    func=duckduckgo_search,
    args_schema=SearchQuery,
    description=(
        "Searches the web for general tea cultivation information (diseases, pests, fertilizer, weather, practices) "
        "when the local knowledge base is insufficient. Input: a search query (string)."
    ),
)
