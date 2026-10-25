"""A farmer's past fertilization records, newest first, as a readable sentence."""
from datetime import datetime

from agent.db.sqlite.tea_farm_db import tea_farm_db


def fertilize_history_by_user(user_id: int) -> str:
    history = tea_farm_db.get_fertilize_history_by_user(user_id)  # already newest first
    if not history:
        return "No fertilization history found for this farmer."
    entries = [
        f"You fertilized on {datetime.strptime(e['fertilizedDate'], '%Y-%m-%d').strftime('%B %d, %Y')} "
        f"with {e['fertilizedAmount']:g} kg"
        for e in history
    ]
    return ". ".join(entries) + "."
