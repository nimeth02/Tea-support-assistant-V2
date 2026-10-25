"""GetFertilizeHistoryByUser tool: a farmer's past fertilization records."""
from datetime import datetime
from typing import Any, Dict, List, Union

from agent.agents.shared_tools.base import TeaTool
from agent.db.sqlite.tea_farm_db import tea_farm_db


def fertilize_history_by_user(userId) -> Union[str, List[Dict[str, Any]]]:
    """Return the fertilization history for a userId as a readable sentence, newest first."""
    print("_get_fertilize_history_by_user")
    user_id = int(userId)

    history = tea_farm_db.get_fertilize_history_by_user(user_id)  # already newest first
    print(history)
    if not history:
        return [{"error": f"No fertilizer history found for userId {user_id}"}]

    formatted_entries = [
        f"You fertilized on {datetime.strptime(entry['fertilizedDate'], '%Y-%m-%d').strftime('%B %d, %Y')} "
        f"with {entry['fertilizedAmount']:g} kg"
        for entry in history
    ]
    return ". ".join(formatted_entries) + "."


class FertilizeHistoryTool(TeaTool):
    name: str = "GetFertilizeHistoryByUser"
    description: str = "Fetches the farmer's fertilization history (dates and amounts), newest first. Takes no input - it already knows the farmer."

    def _run(self) -> Any:
        self.note_use()
        return fertilize_history_by_user(self.chat_state.user_id)
