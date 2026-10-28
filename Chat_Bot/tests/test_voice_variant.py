"""Tests for the multilingual + voice variant that need no API keys or network.

They cover the pure helpers (language detection, markdown stripping, transcription parsing) and
the variant registry. Run from Chat_Bot:  .venv\\Scripts\\python -m pytest tests -q
"""
import pytest

from agent_loader import mount_agent_variant
from variants import ROOT, VARIANTS

mount_agent_variant("voice")  # make "agent multilangual and voice also" importable as `agent`

from agent.services.language_service import (  # noqa: E402
    LANGUAGES, detect_script, normalize_language, resolve_language,
)
from agent.services.speech_service import parse_transcription, strip_markdown  # noqa: E402


# ------------------------------------------------------------------ variant registry
def test_voice_variant_is_registered_with_its_own_ports():
    voice = VARIANTS["voice"]
    assert voice.folder == "agent multilangual and voice also"
    assert voice.path.is_dir()
    assert (ROOT / "frontend" / "app_voice.py").is_file()
    backend_ports = [v.backend_port for v in VARIANTS.values()]
    ui_ports = [v.ui_port for v in VARIANTS.values()]
    assert len(set(backend_ports)) == len(backend_ports)
    assert len(set(ui_ports)) == len(ui_ports)


def test_every_variant_has_a_ui_file():
    for v in VARIANTS.values():
        assert (ROOT / "frontend" / f"app_{v.key}.py").is_file(), v.key


# ------------------------------------------------------------------ language detection
@pytest.mark.parametrize("text, expected", [
    ("මගේ තේ කොළවල දුඹුරු පැල්ලම් තියෙනවා", "si"),
    ("நான் கடைசியாக எப்போது உரம் இட்டேன்?", "ta"),
    ("How much fertilizer should I use?", "en"),
    ("mage the kola kaha wela", "en"),          # romanized Sinhala looks like English to the detector
    ("Urea 50kg - මොකද කරන්නේ?", "si"),         # mixed: the Sinhala script wins over Latin letters
    ("", "en"),
])
def test_detect_script(text, expected):
    assert detect_script(text) == expected


@pytest.mark.parametrize("value, expected", [
    ("si", "si"), ("Sinhala", "si"), ("si-LK", "si"),
    ("ta", "ta"), ("TAMIL", "ta"),
    ("en", "en"), ("English", "en"),
    ("auto", "auto"), ("", "auto"), (None, "auto"),
    ("fr", None), ("klingon", None),
])
def test_normalize_language(value, expected):
    assert normalize_language(value) == expected


def test_resolve_language_prefers_explicit_choice_over_script():
    sinhala = "මගේ තේ කොළ"
    assert resolve_language("auto", sinhala) == "si"
    assert resolve_language(None, sinhala) == "si"
    assert resolve_language("ta", sinhala) == "ta"         # the user's choice wins
    assert resolve_language("si", "mage the kola") == "si"  # romanized text + explicit language


def test_language_table_is_consistent():
    assert set(LANGUAGES) == {"en", "si", "ta"}
    for code, lang in LANGUAGES.items():
        assert lang.code == code
        assert lang.as_dict() == {"code": code, "name": lang.name, "native": lang.native}


# ------------------------------------------------------------------ speech helpers
def test_strip_markdown_leaves_plain_speech_text():
    text = ("## Blister blight\n\n"
            "Your leaves show **blister blight** (*Exobasidium vexans*).\n"
            "- Spray `copper fungicide` every 7 days\n"
            "1. Remove infected leaves\n"
            "See [the guide](http://example.com) for more.")
    spoken = strip_markdown(text)
    for marker in ("#", "**", "*", "`", "- ", "1. ", "http://", "["):
        assert marker not in spoken, marker
    assert "Blister blight" in spoken
    assert "copper fungicide" in spoken
    assert "the guide" in spoken


@pytest.mark.parametrize("raw, text, language", [
    ('{"language": "si", "text": "මගේ තේ කොළ"}', "මගේ තේ කොළ", "si"),
    ('```json\n{"language": "ta", "text": "உரம்"}\n```', "உரம்", "ta"),
    ('Sure! {"language": "English", "text": "hello there"}', "hello there", "en"),
    ('{"language": "other", "text": ""}', "", None),
    ("just the words, no json", "just the words, no json", None),
])
def test_parse_transcription(raw, text, language):
    assert parse_transcription(raw) == (text, language)
