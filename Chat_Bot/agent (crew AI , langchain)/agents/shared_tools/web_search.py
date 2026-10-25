"""Web search fallback (DuckDuckGo) for when a local knowledge base has no answer. Framework-free."""
import logging

from duckduckgo_search import DDGS

logger = logging.getLogger("agent.tools.web_search")


def duckduckgo_search(query: str) -> str:
    """Return up to three web results as a bullet list of title and link."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region="in-en", safesearch="moderate", max_results=3)
    except Exception as e:
        logger.warning("web_search_failed error=%s", e)
        return f"Web search failed: {e}"
    if not results:
        return "No results found from the web."
    return "\n\n".join(f"- {r['title']}: {r['href']}" for r in results)
