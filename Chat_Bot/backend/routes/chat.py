import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image

from backend.models.chat import ChatResponse
from agent.agents.orchestrator.orchestrator_agent import OrchestratorAgent
from agent_loader import active_variant
from observability import ObservabilityCallback, langsmith_tracing, metrics, store

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_TYPES = {"image/jpeg": ".jpg", "image/png": ".png"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MB

DEFAULT_IMAGE_QUERY = "Please identify the disease or pest affecting this tea leaf from the attached photo."


@router.get("/welcome")
async def read_root():
    return {"message": "Welcome to the Chat API"}


@router.get("/variant")
async def variant_info():
    """Which agent variant this backend runs, and where its LangSmith traces go (for the UI)."""
    variant = active_variant()
    info = {
        "key": variant.key,
        "title": variant.title,
        "framework": variant.framework,
        "langsmith_enabled": langsmith_tracing.tracing_enabled(),
        "langsmith_project": variant.langsmith_project,
        "langsmith_project_url": langsmith_tracing.project_url(variant),
    }
    if variant.key in ("a2a", "hybrid"):
        from agent.config import settings
        info["a2a_agent_urls"] = settings.a2a_agent_urls
    return info


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


@router.post("/chat", response_model=ChatResponse)
async def chat(
    http_request: Request,
    message: str = Form(default=""),
    image: Optional[UploadFile] = File(default=None),
):
    trace_id = http_request.state.trace_id
    variant = active_variant()
    userId = 3
    start = time.perf_counter()
    intent, response_text, status, error = "error", "", "ok", None
    try:
        image_path = await _save_uploaded_image(image, trace_id) if image else None
        query = message.strip() or (DEFAULT_IMAGE_QUERY if image_path else "")
        if not query:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        orchestrator_agent = OrchestratorAgent()
        # Root LangSmith run for this chat; everything the agent does is traced underneath it.
        with langsmith_tracing.chat_run(variant, trace_id, query, userId, image_path) as ls_run:
            response = await orchestrator_agent.process_chat(
                query, userId, image_path=image_path, callbacks=[ObservabilityCallback(trace_id)]
            )
            if ls_run is not None:
                ls_run.end(outputs=response)
        print(response)
        intent = response.get("intent") or "unknown"
        response_text = response.get("response", "")
        if intent == "error":
            status = "error"
        return {**response, "trace_id": trace_id, "variant": variant.key,
                "langsmith_url": langsmith_tracing.run_url(variant, ls_run.id) if ls_run else None,
                "latency_ms": round((time.perf_counter() - start) * 1000)}
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
        store.save_trace(trace_id, userId, message, intent,
                         response_text, status, error, latency_ms)
