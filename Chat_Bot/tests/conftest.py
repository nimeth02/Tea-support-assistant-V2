"""Pytest setup for Chat_Bot tests: run them from the Chat_Bot folder (`python -m pytest tests`).

Makes Chat_Bot/ importable (variants.py, agent_loader.py, backend/, frontend/) whatever folder
pytest was started from.
"""
import os
import sys
from pathlib import Path

CHAT_BOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHAT_BOT))
os.chdir(CHAT_BOT)  # agent/config.py reads .env relative to the working directory
