"""Health checks for the expert servers.

Two kinds, the standard split used by Docker and Kubernetes:

  liveness  (/healthz)  "Is the process alive?" Always cheap. If it fails, restart the container.
  readiness (/readyz)   "Can it do real work right now?" Checks the things a question needs: an
                        LLM key, the ML model file, the database, the vector store. If it fails,
                        stop sending traffic to it, but don't restart it (restarting won't fix
                        a missing API key).

Each check is a small function that returns (ok, detail). Each expert picks the checks it needs.
"""
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from agent.config import settings

Check = Callable[[], Tuple[bool, str]]


def llm_configured() -> Tuple[bool, str]:
    if settings.LLM_PROVIDER.lower() == "gemini":
        return bool(settings.GEMINI_API_KEY), f"gemini ({settings.GEMINI_MODEL})"
    return bool(settings.GROQ_API_KEY), f"groq ({settings.GROQ_MODEL})"


def file_exists(path: str, label: str) -> Check:
    def check() -> Tuple[bool, str]:
        return Path(path).is_file(), label
    check.__name__ = label
    return check


def tea_farm_db_readable() -> Tuple[bool, str]:
    try:
        from agent.db.sqlite.tea_farm_db import tea_farm_db
        tea_farm_db.get_session().execute("SELECT 1").fetchone()
        return True, "tea_farm.db"
    except Exception as e:
        return False, f"tea_farm.db: {e}"


def vector_db_ready(kb: str) -> Check:
    """The knowledge base exists. For Chroma: the collection is on disk and not empty."""
    def check() -> Tuple[bool, str]:
        if settings.VECTOR_DB_PROVIDER.lower() != "chroma":
            ok = bool(settings.PINECONE_API_KEY and settings.COHERE_API_KEY)
            return ok, f"pinecone {kb} (keys set)"
        try:
            import chromadb
            from agent.db.vector.chroma.chroma_vector_db import CHROMA_PATH
            name = settings.CHROMA_DISEASE_COLLECTION if kb == "disease" else settings.CHROMA_HELP_COLLECTION
            count = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(name).count()
            return count > 0, f"chroma {name}: {count} records"
        except Exception as e:
            return False, f"chroma {kb}: {e}"
    check.__name__ = f"vector_db_{kb}"
    return check


def run_checks(checks: List[Check]) -> Tuple[bool, Dict[str, Dict]]:
    results = {}
    for check in checks:
        try:
            ok, detail = check()
        except Exception as e:  # a broken check must not crash the endpoint
            ok, detail = False, str(e)
        results[check.__name__] = {"ok": ok, "detail": detail}
    return all(r["ok"] for r in results.values()), results
