"""Input validation for the disease classifier.

Same pattern as agent/agents/fertilize/tools/schemas.py: catch invalid
input at the service boundary, before it reaches the model. Pydantic can't
check pixel content - that's still PIL's job (Image.open) - but it catches
the cheap, common failure modes before touching the file at all: an empty
path, an unsupported file type, a missing file, or something absurdly
large, each of which deserves a clear error message instead of a PIL/torch
traceback several layers deep.
"""
from pathlib import Path

from pydantic import BaseModel, field_validator

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_FILE_SIZE_MB = 20


class ImageInput(BaseModel):
    """The image classifier's input, validated before PIL ever opens the file."""

    image_path: str

    @field_validator("image_path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("image path is empty")

        path = Path(v)
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"unsupported file type {path.suffix!r} (expected one of {sorted(ALLOWED_EXTENSIONS)})"
            )
        if not path.exists():
            raise ValueError(f"file does not exist: {v}")

        size_bytes = path.stat().st_size
        if size_bytes == 0:
            raise ValueError("file is empty")
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(f"file too large ({size_mb:.1f} MB > {MAX_FILE_SIZE_MB} MB limit)")

        return v
