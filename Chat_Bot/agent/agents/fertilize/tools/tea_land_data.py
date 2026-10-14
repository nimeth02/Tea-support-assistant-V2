"""TeaLandDataByUser tool: a farmer's tea land records (location, size, slope, ...)."""
from typing import Any, Dict, List

from langchain.tools import Tool

from agent.db.sqlite.tea_farm_db import tea_farm_db


def tea_land_data_by_user(userId) -> List[Dict[str, Any]]:
    """Return the tea land records registered for a userId."""
    print("_tea_land_data_by_user")
    user_id = int(userId)

    result = tea_farm_db.get_tea_land_by_user(user_id)
    print(result)
    if not result:
        return [{"error": f"No data found for userId {user_id}"}]
    return result


tea_land_data_tool = Tool(
    name="TeaLandDataByUser",
    func=tea_land_data_by_user,
    description="Retrieves tea land data (e.g., location, land size) for a userId. Input: userId (integer, e.g., 3).",
)
