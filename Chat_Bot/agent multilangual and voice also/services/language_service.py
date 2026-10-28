"""Language detection and translation for the multilingual variant.

The experts, their prompts and the RAG knowledge bases only work in English. This service
sits around them: it works out which language the farmer used, turns the question into
English, and later turns the English answer back into the farmer's language.

Detection is done *without* an LLM call: Sinhala and Tamil each have their own Unicode script
(their own block of character codes), so counting characters tells us the language for free.
Latin-letter text is treated as English. A farmer typing romanized Sinhala/Tamil ("mage the
kola...") should pick their language in the UI instead - then we translate it regardless.
"""
from dataclasses import dataclass
from typing import Dict, Optional

from agent.prompt_templates.translate_prompt import (
    translate_from_english_prompt,
    translate_to_english_prompt,
)
from agent.services.llm_service import LLMService

AUTO = "auto"


@dataclass(frozen=True)
class Language:
    code: str          # ISO 639-1 code used everywhere in the API ("si", "ta", "en")
    name: str          # English name, used in prompts
    native: str        # how speakers write the language's name, shown in the UI
    gtts: str          # gTTS language code
    edge_voice: str    # Microsoft Edge neural voice name
    whisper: str       # the name Whisper's verbose_json reports (lowercase)

    def as_dict(self) -> Dict[str, str]:
        return {"code": self.code, "name": self.name, "native": self.native}


LANGUAGES: Dict[str, Language] = {
    lang.code: lang
    for lang in [
        Language("en", "English", "English", "en", "en-GB-SoniaNeural", "english"),
        Language("si", "Sinhala", "සිංහල", "si", "si-LK-ThiliniNeural", "sinhala"),
        Language("ta", "Tamil", "தமிழ்", "ta", "ta-LK-SaranyaNeural", "tamil"),
    ]
}

# Unicode blocks of the non-Latin scripts we support (start, end code point, inclusive).
SCRIPT_RANGES = {
    "si": (0x0D80, 0x0DFF),  # Sinhala
    "ta": (0x0B80, 0x0BFF),  # Tamil
}

_ALIASES = {
    "auto": AUTO, "": AUTO, "detect": AUTO,
    "en": "en", "eng": "en", "english": "en", "en-us": "en", "en-gb": "en",
    "si": "si", "sin": "si", "sinhala": "si", "si-lk": "si", "sinhalese": "si",
    "ta": "ta", "tam": "ta", "tamil": "ta", "ta-lk": "ta", "ta-in": "ta",
}


def normalize_language(value: Optional[str]) -> Optional[str]:
    """"Sinhala" / "si-LK" / "si" -> "si"; "auto"/empty -> "auto"; anything else -> None."""
    return _ALIASES.get((value or "").strip().lower())


def detect_script(text: str) -> str:
    """Language code guessed from the characters used: "si", "ta", or "en" for everything else."""
    counts = {code: 0 for code in SCRIPT_RANGES}
    for ch in text or "":
        cp = ord(ch)
        for code, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[code] += 1
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else "en"


def resolve_language(requested: Optional[str], text: str) -> str:
    """The language to work in: what the user asked for, or detected from the text on "auto"."""
    code = normalize_language(requested)
    if code is None or code == AUTO:
        return detect_script(text)
    return code


class LanguageService:
    """Translates between English and the farmer's language with the shared LLM."""

    def __init__(self):
        self.llm = LLMService().get_llm()

    async def to_english(self, text: str, language: str) -> str:
        """Farmer's message -> English. Returns the text unchanged for English or on failure."""
        if language == "en" or not text.strip():
            return text
        return await self._translate(translate_to_english_prompt, text, language)

    async def from_english(self, text: str, language: str) -> str:
        """English answer -> farmer's language. Returns the text unchanged for English or on failure."""
        if language == "en" or not text.strip():
            return text
        return await self._translate(translate_from_english_prompt, text, language)

    async def _translate(self, prompt, text: str, language: str) -> str:
        name = LANGUAGES[language].name if language in LANGUAGES else language
        try:
            response = await self.llm.ainvoke(prompt.format(language=name, text=text))
            translated = (response.text or "").strip()
            return translated or text
        except Exception as e:  # a failed translation should never lose the answer itself
            print(f"Translation to/from {name} failed, keeping original text: {e}")
            return text
