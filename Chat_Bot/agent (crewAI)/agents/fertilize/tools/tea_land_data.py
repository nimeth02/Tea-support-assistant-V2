"""TeaLandDataByUser tool: a farmer's tea land records (location, size, slope, ...)."""
from typing import Any, Dict, List

from agent.agents.shared_tools.base import TeaTool
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


class TeaLandDataTool(TeaTool):
    name: str = "TeaLandDataByUser"
    description: str = "Retrieves the farmer's tea land data (location, land size, slope, number of plants). Takes no input - it already knows the farmer."

    def _run(self) -> Any:
        self.note_use()
        return tea_land_data_by_user(self.chat_state.user_id)
