"""Guardrails: cheap, rule-based checks on what goes into and comes out of the LLMs.

They are a first line of defence, not a complete one. They run in microseconds and catch the
common, obvious cases before any LLM call is paid for:

  check_input     empty or oversized messages, and well-known prompt-injection phrases
                  ("ignore all previous instructions", "reveal your system prompt", ...).
                  Prompt injection = text that tries to make the model drop its instructions.
  clean_output    removes things a farmer must never see: local file paths (where an uploaded
                  photo was saved) and anything that looks like an API key. Also caps the length.

Every block / clean-up is counted in the guardrail_events_total metric, so a sudden spike (for
example, someone probing the bot) shows up on the dashboard and can raise an alert.
"""
import re
from dataclasses import dataclass
from typing import Optional

from agent.config import settings
from agent.ops.telemetry import GUARDRAIL_EVENTS

BLOCKED_RESPONSE = (
    "Sorry, I can't help with that request. I can answer questions about tea fertilizing, "
    "tea diseases and pests, and the tea company's services."
)

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)",
    r"disregard\s+(all\s+|your\s+)?(previous\s+|prior\s+)?(instructions|rules)",
    r"(reveal|show|print|repeat)\s+(me\s+)?(your|the)\s+(system\s+prompt|instructions|hidden\s+prompt)",
    r"you\s+are\s+now\s+(a|an|in)\s+",
    r"\bjailbreak\b",
    r"\bdeveloper\s+mode\b",
    r"act\s+as\s+(an?\s+)?(unfiltered|uncensored)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# Things that must never reach the farmer.
_SECRET_RE = re.compile(r"\b(AIza[0-9A-Za-z_\-]{20,}|gsk_[0-9A-Za-z]{20,}|sk-[0-9A-Za-z]{20,})\b")
_PATH_RE = re.compile(
    r"([A-Za-z]:\\[^\s'\"<>]+|(?:\.{0,2}/)?(?:data|tmp|app|home|Users)/[^\s'\"<>]*uploads?/[^\s'\"<>]+)"
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class InputCheck:
    allowed: bool
    text: str = ""
    reason: Optional[str] = None
    message: Optional[str] = None  # what to tell the farmer when not allowed


def check_input(text: str, max_chars: int = None) -> InputCheck:
    """Validate and normalise one farmer message before any agent sees it."""
    max_chars = max_chars or settings.MAX_INPUT_CHARS
    cleaned = _CONTROL_RE.sub("", text or "").strip()

    if not cleaned:
        return _block("empty", "Please type a question about your tea plants or the tea company.")
    if len(cleaned) > max_chars:
        return _block(
            "too_long",
            f"Your message is too long. Please keep it under {max_chars} characters.",
        )
    if _INJECTION_RE.search(cleaned):
        return _block("prompt_injection", BLOCKED_RESPONSE)
    return InputCheck(allowed=True, text=cleaned)


def clean_output(text: str, max_chars: int = None) -> str:
    """Remove file paths and secret-looking strings, and cap the length of a reply."""
    max_chars = max_chars or settings.MAX_OUTPUT_CHARS
    text = text or ""

    if _SECRET_RE.search(text):
        GUARDRAIL_EVENTS.labels("output", "secret").inc()
        text = _SECRET_RE.sub("[removed]", text)
    if _PATH_RE.search(text):
        GUARDRAIL_EVENTS.labels("output", "file_path").inc()
        text = _PATH_RE.sub("[your photo]", text)
    if len(text) > max_chars:
        GUARDRAIL_EVENTS.labels("output", "too_long").inc()
        text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
    return text.strip()


def _block(reason: str, message: str) -> InputCheck:
    GUARDRAIL_EVENTS.labels("input", reason).inc()
    return InputCheck(allowed=False, reason=reason, message=message)
