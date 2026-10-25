"""SQLite store for chat traces and their spans (LLM calls, tool calls, graph nodes)."""
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# <project root>/data/observability.db for the default "agent" variant, and
# observability_<variant>.db for the others, so each variant's dashboard shows only its own chats.
_VARIANT = os.getenv("AGENT_VARIANT", "agent")
DB_PATH = Path(__file__).resolve().parents[1] / "data" / (
    "observability.db" if _VARIANT == "agent" else f"observability_{_VARIANT}.db")
_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock, _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS traces (
                trace_id    TEXT PRIMARY KEY,
                ts          REAL NOT NULL,
                user_id     INTEGER,
                message     TEXT,
                intent      TEXT,
                response    TEXT,
                status      TEXT,
                error       TEXT,
                latency_ms  REAL
            );
            CREATE TABLE IF NOT EXISTS spans (
                span_id       TEXT PRIMARY KEY,
                trace_id      TEXT NOT NULL,
                parent_id     TEXT,
                kind          TEXT,
                name          TEXT,
                model         TEXT,
                start_ts      REAL,
                latency_ms    REAL,
                status        TEXT,
                input         TEXT,
                output        TEXT,
                error         TEXT,
                input_tokens  INTEGER,
                output_tokens INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
            CREATE INDEX IF NOT EXISTS idx_traces_ts ON traces(ts);
            """
        )


def _truncate(value: Any, limit: int = 2000) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


def save_trace(trace_id: str, user_id: int, message: str, intent: str,
               response: str, status: str, error: Optional[str], latency_ms: float) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?,?,?)",
            (trace_id, time.time(), user_id, _truncate(message), intent,
             _truncate(response), status, _truncate(error), latency_ms),
        )


def save_span(span: Dict[str, Any]) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO spans
               (span_id, trace_id, parent_id, kind, name, model, start_ts, latency_ms,
                status, input, output, error, input_tokens, output_tokens)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (span["span_id"], span["trace_id"], span.get("parent_id"), span["kind"],
             span["name"], span.get("model"), span["start_ts"], span["latency_ms"],
             span["status"], _truncate(span.get("input")), _truncate(span.get("output")),
             _truncate(span.get("error")), span.get("input_tokens"), span.get("output_tokens")),
        )


def list_traces(limit: int = 100) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT t.*,
                      (SELECT COUNT(*) FROM spans s WHERE s.trace_id = t.trace_id AND s.kind = 'llm') AS llm_calls,
                      (SELECT COUNT(*) FROM spans s WHERE s.trace_id = t.trace_id AND s.kind = 'tool') AS tool_calls,
                      (SELECT COALESCE(SUM(input_tokens), 0) + COALESCE(SUM(output_tokens), 0)
                         FROM spans s WHERE s.trace_id = t.trace_id) AS total_tokens
               FROM traces t ORDER BY ts DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_trace(trace_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        trace = conn.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
        if not trace:
            return None
        spans = conn.execute(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_ts", (trace_id,)
        ).fetchall()
    return {**dict(trace), "spans": [dict(s) for s in spans]}


def list_spans(kind: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
    query = "SELECT * FROM spans"
    params: tuple = ()
    if kind:
        query += " WHERE kind = ?"
        params = (kind,)
    query += " ORDER BY start_ts DESC LIMIT ?"
    with _connect() as conn:
        rows = conn.execute(query, params + (limit,)).fetchall()
    return [dict(r) for r in rows]


def stats() -> Dict[str, Any]:
    with _connect() as conn:
        t = conn.execute(
            """SELECT COUNT(*) AS total,
                      SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END) AS errors,
                      AVG(latency_ms) AS avg_latency_ms
               FROM traces"""
        ).fetchone()
        latencies = [r[0] for r in conn.execute(
            "SELECT latency_ms FROM traces ORDER BY latency_ms").fetchall()]
        s = conn.execute(
            """SELECT SUM(CASE WHEN kind='llm' THEN 1 ELSE 0 END) AS llm_calls,
                      SUM(CASE WHEN kind='llm' AND status!='ok' THEN 1 ELSE 0 END) AS llm_errors,
                      SUM(CASE WHEN kind='tool' THEN 1 ELSE 0 END) AS tool_calls,
                      COALESCE(SUM(input_tokens), 0) AS input_tokens,
                      COALESCE(SUM(output_tokens), 0) AS output_tokens
               FROM spans"""
        ).fetchone()
        intents = conn.execute(
            "SELECT intent, COUNT(*) AS count FROM traces GROUP BY intent ORDER BY count DESC"
        ).fetchall()

    def pct(p: float) -> Optional[float]:
        if not latencies:
            return None
        return latencies[min(len(latencies) - 1, int(p * len(latencies)))]

    return {
        "total_requests": t["total"] or 0,
        "errors": t["errors"] or 0,
        "avg_latency_ms": t["avg_latency_ms"],
        "p50_latency_ms": pct(0.5),
        "p95_latency_ms": pct(0.95),
        **{k: (s[k] or 0) for k in s.keys()},
        "intents": {r["intent"] or "unknown": r["count"] for r in intents},
    }
