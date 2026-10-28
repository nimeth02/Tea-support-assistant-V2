import inspect
import logging
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from backend.models.chat import ChatResponse
from agent.agents.orchestrator.orchestrator_agent import OrchestratorAgent
from agent_loader import active_variant
from observability import ObservabilityCallback, langsmith_tracing, metrics, store

router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR = Path("data/audio")  # spoken answers (MP3), one per chat, served by GET /audio/{trace_id}

ALLOWED_IMAGE_TYPES = {"image/jpeg": ".jpg", "image/png": ".png"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MB

# What browsers record (Streamlit's microphone gives WAV) plus common audio uploads.
ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave", "audio/webm", "audio/ogg", "audio/mpeg",
    "audio/mp3", "audio/mp4", "audio/m4a", "audio/x-m4a", "audio/aac", "audio/flac",
}
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB (~ a few minutes of speech)

DEFAULT_IMAGE_QUERY = "Please identify the disease or pest affecting this tea leaf from the attached photo."


# ---------------------------------------------------------------------------------------------
# Voice + language support. Only the "multilingual + voice" variant ships these services; for the
# other variants the imports fail and the backend simply behaves as before (English, text only).
# ---------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def speech_service():
    """The variant's SpeechService (speech-to-text and text-to-speech), or None without voice support."""
    try:
        from agent.services.speech_service import SpeechService
    except ImportError:
        return None
    return SpeechService()


def supported_languages() -> List[dict]:
    """[{"code", "name", "native"}, ...] the running variant can answer in."""
    try:
        from agent.services.language_service import LANGUAGES
    except ImportError:
        return [{"code": "en", "name": "English", "native": "English"}]
    return [lang.as_dict() for lang in LANGUAGES.values()]


def accepts_language(orchestrator: OrchestratorAgent) -> bool:
    """Whether this variant's process_chat() takes a `language` argument."""
    return "language" in inspect.signature(orchestrator.process_chat).parameters


def _check_language(value: str) -> str:
    code = (value or "auto").strip().lower()
    allowed = {"auto"} | {lang["code"] for lang in supported_languages()}
    if code not in allowed:
        raise HTTPException(status_code=400,
                            detail=f"Unsupported language {value!r}. Choose one of: {', '.join(sorted(allowed))}.")
    return code


@router.get("/welcome")
async def read_root():
    return {"message": "Welcome to the Chat API"}


@router.get("/variant")
async def variant_info():
    """Which agent variant this backend runs, and where its LangSmith traces go (for the UI)."""
    variant = active_variant()
    speech = speech_service()
    info = {
        "key": variant.key,
        "title": variant.title,
        "framework": variant.framework,
        "langsmith_enabled": langsmith_tracing.tracing_enabled(),
        "langsmith_project": variant.langsmith_project,
        "langsmith_project_url": langsmith_tracing.project_url(variant),
        "features": {"voice": speech is not None, "languages": supported_languages()},
    }
    if speech is not None:
        info["speech"] = speech.info()  # stt_provider, tts_provider
    if variant.key in ("a2a", "hybrid"):
        from agent.config import settings
        info["a2a_agent_urls"] = settings.a2a_agent_urls
    return info


@router.get("/languages")
async def languages():
    """Languages this backend can answer in. "auto" means: detect from the message."""
    return {"default": "auto", "languages": supported_languages()}


async def _save_uploaded_image(image: UploadFile, trace_id: str) -> str:
    """Validate an uploaded leaf photo (type, size, that it's actually an image) and save it."""
    suffix = ALLOWED_IMAGE_TYPES.get(image.content_type)
    if suffix is None:
        raise HTTPException(status_code=400, detail="Only JPEG or PNG images are supported.")

    contents = await image.read()
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image is too large (max 8 MB).")

    image_path = UPLOAD_DIR / f"{trace_id}{suffix}"
    image_path.write_bytes(contents)

    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception:
        image_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Could not read the image file - it may be corrupted.")

    return str(image_path)


async def _transcribe(speech, audio: UploadFile, language: str) -> Tuple[str, Optional[str]]:
    """Validate a voice recording and turn it into text. Returns (words heard, language heard)."""
    mime_type = (audio.content_type or "audio/wav").split(";")[0].strip().lower()
    if mime_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported audio type {mime_type!r}. Record in WAV, WebM, OGG, MP3 or M4A.")

    contents = await audio.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The recording is empty.")
    if len(contents) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=400, detail="Recording is too long (max 10 MB).")

    try:
        return await speech.transcribe(contents, mime_type, language)
    except Exception as e:
        logger.exception("Speech recognition failed")
        raise HTTPException(status_code=502, detail=f"Speech recognition failed: {e}")


async def _speak(speech, text: str, language: str, trace_id: str) -> Optional[str]:
    """Save the answer as MP3 and return its URL path; None if text-to-speech failed (the text
    answer is still returned, speech is a bonus)."""
    try:
        data = await speech.synthesize(text, language)
    except Exception as e:
        logger.warning("Text-to-speech failed, returning text only: %s", e)
        return None
    if not data:
        return None
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIO_DIR / f"{trace_id}.mp3").write_bytes(data)
    return f"/audio/{trace_id}"


@router.get("/audio/{trace_id}")
async def audio_reply(trace_id: str):
    """The spoken answer (MP3) for a chat, when it was requested with speak=true."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", trace_id):
        raise HTTPException(status_code=404, detail="No audio for that trace id.")
    path = AUDIO_DIR / f"{trace_id}.mp3"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No audio for that trace id.")
    return FileResponse(path, media_type="audio/mpeg", filename=f"{trace_id}.mp3")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    http_request: Request,
    message: str = Form(default=""),
    image: Optional[UploadFile] = File(default=None),
    audio: Optional[UploadFile] = File(default=None),   # a voice recording of the question
    language: str = Form(default="auto"),               # "auto", or a code from GET /languages
    speak: bool = Form(default=False),                  # also return the answer as speech
):
    trace_id = http_request.state.trace_id
    variant = active_variant()
    userId = 3
    start = time.perf_counter()
    intent, response_text, status, error = "error", "", "ok", None
    query = message.strip()
    try:
        speech = speech_service()
        language = _check_language(language)
        image_path = await _save_uploaded_image(image, trace_id) if image else None

        transcript, heard_language = None, None
        if audio is not None:
            if speech is None:
                raise HTTPException(status_code=400,
                                    detail=f"The {variant.title} variant does not support voice messages. "
                                           f"Start the voice variant with: python launch.py up voice")
            transcript, heard_language = await _transcribe(speech, audio, language)
            query = f"{query}\n{transcript}".strip() if query else transcript
        if language == "auto" and heard_language:
            language = heard_language  # answer in the language the farmer spoke

        query = query or (DEFAULT_IMAGE_QUERY if image_path else "")
        if not query:
            raise HTTPException(status_code=400,
                                detail="I couldn't hear any words in that recording - please try again."
                                       if audio is not None else "Message cannot be empty.")

        orchestrator_agent = OrchestratorAgent()
        extra = {"language": language} if accepts_language(orchestrator_agent) else {}
        # Root LangSmith run for this chat; everything the agent does is traced underneath it.
        with langsmith_tracing.chat_run(variant, trace_id, query, userId, image_path) as ls_run:
            response = await orchestrator_agent.process_chat(
                query, userId, image_path=image_path, callbacks=[ObservabilityCallback(trace_id)], **extra
            )
            if ls_run is not None:
                ls_run.end(outputs=response)
        print(response)
        intent = response.get("intent") or "unknown"
        response_text = response.get("response", "")
        if intent == "error":
            status = "error"

        result = {**response, "trace_id": trace_id, "variant": variant.key, "transcript": transcript,
                  "langsmith_url": langsmith_tracing.run_url(variant, ls_run.id) if ls_run else None}
        if speak and speech is not None and response_text and intent != "error":
            result["audio_url"] = await _speak(speech, response_text,
                                               response.get("language") or heard_language or "en", trace_id)
        result["latency_ms"] = round((time.perf_counter() - start) * 1000)
        return result
    except HTTPException as he:
        status, error = "error", str(he.detail)
        raise
    except Exception as e:
        status, error = "error", str(e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        metrics.CHAT_REQUESTS.labels(intent, status).inc()
        metrics.CHAT_LATENCY.labels(intent).observe(latency_ms / 1000)
        store.save_trace(trace_id, userId, query or message, intent,
                         response_text, status, error, latency_ms)
