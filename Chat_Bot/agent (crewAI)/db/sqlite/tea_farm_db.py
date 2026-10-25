"""Read-only SQLite access to farmer reference data (tea land + fertilization history).

The database file ships with the code (agent/db/data/tea_farm.db). The agent only ever
reads from it, so every connection is opened read-only:

  * ``mode=ro`` in the SQLite URI makes the connection itself read-only. Any
    INSERT/UPDATE/DELETE/CREATE raises ``sqlite3.OperationalError:
    attempt to write a readonly database``. The file is never created or grown.
  * ``PRAGMA query_only = ON`` is a second guard at the statement level, so even
    if someone later swaps the URI for a plain path the writes still fail.

Tables
  tea_land(user_id PK, location, land_size, plant_amount, degree)
  fertilize_history(id PK, user_id -> tea_land, fertilized_amount, fertilized_date)
"""
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.db.base_db import BaseDatabase

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "tea_farm.db"


def connect_readonly(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open a read-only connection to the SQLite file."""
    if not db_path.exists():
        # With mode=ro SQLite would report "unable to open database file"; be explicit.
        raise FileNotFoundError(f"Tea farm database not found: {db_path}")
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


class TeaFarmDatabase(BaseDatabase):
    """Read-only repository for tea land records and fertilization history."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self.connect()

    # ---- BaseDatabase ---------------------------------------------------
    def connect(self) -> None:
        self._conn = connect_readonly(self.db_path)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def get_session(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        return self._conn

    # ---- queries ----------------------------------------------------------
    def get_tea_land_by_user(self, user_id: int) -> List[Dict[str, Any]]:
        """Tea land records for a farmer, keyed the way the agent tools/prompts expect."""
        rows = self.get_session().execute(
            """SELECT user_id      AS userId,
                      location,
                      land_size    AS landSize,
                      plant_amount AS plantAmount,
                      degree
               FROM tea_land
               WHERE user_id = ?""",
            (int(user_id),),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_fertilize_history_by_user(self, user_id: int) -> List[Dict[str, Any]]:
        """Fertilization records for a farmer, newest first."""
        rows = self.get_session().execute(
            """SELECT user_id           AS userId,
                      fertilized_amount AS fertilizedAmount,
                      fertilized_date   AS fertilizedDate
               FROM fertilize_history
               WHERE user_id = ?
               ORDER BY fertilized_date DESC""",
            (int(user_id),),
        ).fetchall()
        return [dict(r) for r in rows]


# One shared read-only connection for the agent tools
tea_farm_db = TeaFarmDatabase()
