"""A farmer's tea land records (location, size, slope, number of plants)."""
from typing import Any, Dict, List

from agent.db.sqlite.tea_farm_db import tea_farm_db


def tea_land_data_by_user(user_id: int) -> List[Dict[str, Any]]:
    rows = tea_farm_db.get_tea_land_by_user(user_id)
    return rows or [{"error": "No tea land is registered for this farmer."}]
