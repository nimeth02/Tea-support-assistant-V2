"""Speech in and speech out for the voice variant.

Two jobs:
  transcribe()  speech-to-text (STT): a recorded question (audio bytes) -> the words spoken,
                plus the language heard. Provider picked by STT_PROVIDER:
                  gemini - the same Gemini chat model, which can listen to audio directly
                  groq   - Groq's hosted Whisper model (OpenAI's open speech recogniser)
  synthesize()  text-to-speech (TTS): the answer -> an MP3 the browser can play.
                Provider picked by TTS_PROVIDER:
                  gtts - Google Translate's voices (gTTS library, no key)
                  edge - Microsoft Edge's neural voices (edge-tts library, no key)

Both providers are only imported when used, so a missing optional library does not stop the
rest of the agent from starting.
"""
import asyncio
import base64
import io
import json
import re
from typing import Optional, Tuple

from langchain_core.messages import HumanMessage

from agent.config import settings
from agent.prompt_templates.translate_prompt import transcribe_prompt
from agent.services.language_service import LANGUAGES, normalize_language
from agent.services.llm_service import LLMService

_MARKDOWN_PATTERNS = [
    (re.compile(r"```.*?```", re.S), " "),              # code blocks
    (re.compile(r"`([^`]*)`"), r"\1"),                   # inline code
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), " "),          # images
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),       # links -> their text
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.M), ""),        # headings
    (re.compile(r"^\s*[-*+]\s+", re.M), ""),             # bullet markers
    (re.compile(r"^\s*\d+[.)]\s+", re.M), ""),           # numbered-list markers
    (re.compile(r"(\*\*|__|\*|_|~~)"), ""),              # bold / italic / strike markers
    (re.compile(r"[ \t]{2,}"), " "),
]


def strip_markdown(text: str) -> str:
    """Plain text for the speaker: no **bold**, # headings, bullets or links."""
    out = text or ""
    for pattern, repl in _MARKDOWN_PATTERNS:
        out = pattern.sub(repl, out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def parse_transcription(raw: str) -> Tuple[str, Optional[str]]:
    """The (text, language code) in the Gemini transcription reply, tolerating code fences
    and plain-text replies from a model that ignored the JSON instruction."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip(), flags=re.S)
    match = re.search(r"\{.*\}", cleaned, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            return (data.get("text") or "").strip(), normalize_language(data.get("language"))
        except (json.JSONDecodeError, AttributeError):
            pass
    return cleaned.strip(), None


class SpeechService:
    def __init__(self):
        self.stt_provider = settings.STT_PROVIDER.lower()
        self.tts_provider = settings.TTS_PROVIDER.lower()
        self._llm = None  # created on first Gemini transcription

    # ------------------------------------------------------------------ speech -> text
    async def transcribe(self, audio: bytes, mime_type: str,
                         language: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """Words spoken in the clip and the language heard ("si"/"ta"/"en" or None if unsure).

        `language` is a hint ("si", "ta", "en" or "auto"/None): when the farmer picked a
        language in the UI we tell the recogniser, which makes it noticeably more accurate.
        """
        hint = normalize_language(language)
        hint = None if hint in (None, "auto") else hint
        if self.stt_provider == "groq":
            return await self._transcribe_groq(audio, mime_type, hint)
        return await self._transcribe_gemini(audio, mime_type, hint)

    async def _transcribe_gemini(self, audio: bytes, mime_type: str, hint: Optional[str]):
        if self._llm is None:
            self._llm = LLMService().get_llm()
        language_hint = (f"3. The farmer said they are speaking {LANGUAGES[hint].name}."
                         if hint in LANGUAGES else "")
        instructions = transcribe_prompt.format(language_hint=language_hint)
        # The audio clip rides along as a second content part, encoded as base64 text.
        message = HumanMessage(content=[
            {"type": "text", "text": instructions},
            {"type": "media", "mime_type": mime_type, "data": base64.b64encode(audio).decode("ascii")},
        ])
        response = await self._llm.ainvoke([message])
        text, heard = parse_transcription(response.text)
        return text, (hint or heard)

    async def _transcribe_groq(self, audio: bytes, mime_type: str, hint: Optional[str]):
        from groq import AsyncGroq  # optional path, imported only when selected

        if not settings.GROQ_API_KEY:
            raise RuntimeError("STT_PROVIDER=groq needs GROQ_API_KEY in .env")
        client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        ext = (mime_type or "audio/wav").split("/")[-1].replace("x-", "").replace("mpeg", "mp3")
        kwargs = {"language": hint} if hint else {}
        result = await client.audio.transcriptions.create(
            file=(f"voice.{ext}", audio, mime_type), model=settings.GROQ_WHISPER_MODEL,
            response_format="verbose_json", temperature=0, **kwargs,
        )
        heard_name = (getattr(result, "language", None) or "").lower()
        heard = next((l.code for l in LANGUAGES.values() if l.whisper == heard_name), None)
        return (result.text or "").strip(), (hint or heard)

    # ------------------------------------------------------------------ text -> speech
    async def synthesize(self, text: str, language: str) -> bytes:
        """MP3 bytes of `text` read aloud in `language` ("en" when the language is unknown)."""
        lang = LANGUAGES.get(language or "en", LANGUAGES["en"])
        spoken = strip_markdown(text)
        if len(spoken) > settings.TTS_MAX_CHARS:
            spoken = spoken[: settings.TTS_MAX_CHARS].rsplit(" ", 1)[0] + " ..."
        if not spoken:
            return b""
        if self.tts_provider == "edge":
            return await self._synthesize_edge(spoken, lang.edge_voice)
        return await self._synthesize_gtts(spoken, lang.gtts)

    @staticmethod
    async def _synthesize_gtts(text: str, lang_code: str) -> bytes:
        from gtts import gTTS

        def run() -> bytes:  # gTTS is blocking (plain HTTP), so run it off the event loop
            buffer = io.BytesIO()
            gTTS(text=text, lang=lang_code).write_to_fp(buffer)
            return buffer.getvalue()

        return await asyncio.to_thread(run)

    @staticmethod
    async def _synthesize_edge(text: str, voice: str) -> bytes:
        import edge_tts

        chunks = []
        async for chunk in edge_tts.Communicate(text, voice).stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)

    # ------------------------------------------------------------------ for the UI
    def info(self) -> dict:
        return {"stt_provider": self.stt_provider, "tts_provider": self.tts_provider,
                "languages": [l.as_dict() for l in LANGUAGES.values()]}
