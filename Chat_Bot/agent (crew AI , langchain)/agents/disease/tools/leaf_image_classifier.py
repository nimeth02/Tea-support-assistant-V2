"""ClassifyLeafImage: predicts the disease/pest in a tea leaf photo with the ResNet18 model
trained in ML Models/Tea Disease Classification/ (path set by DISEASE_MODEL_PATH)."""
import logging
import pickle
import threading

from pydantic import ValidationError

from agent.agents.disease.tools.base import TeaTool
from agent.agents.disease.tools.schemas import ImageInput
from agent.config import settings

logger = logging.getLogger("agent.tools.leaf_classifier")

_bundle = None
_lock = threading.Lock()  # CrewAI may run tools in parallel threads; load the model only once


def _load_bundle() -> dict:
    global _bundle
    with _lock:
        if _bundle is None:
            with open(settings.DISEASE_MODEL_PATH, "rb") as f:
                _bundle = pickle.load(f)
            _bundle["model"].eval()
            logger.info("disease_model_loaded path=%s", settings.DISEASE_MODEL_PATH)
    return _bundle


def classify_leaf_image(image_path: str) -> str:
    """Predict the disease/pest class for the tea leaf photo at image_path."""
    # torch is imported here, not at the top: it takes seconds to load, and the server (and the
    # tests) should start fast even before the first photo arrives.
    import torch
    from PIL import Image
    from torchvision import transforms

    try:
        validated = ImageInput(image_path=image_path)
    except ValidationError as e:
        return f"Invalid image input: {e}"
    try:
        bundle = _load_bundle()
    except Exception as e:
        logger.error("disease_model_load_failed error=%s", e)
        return "The leaf photo model is not available right now."
    try:
        image = Image.open(validated.image_path).convert("RGB")
    except Exception:
        return "Could not open the photo. Please upload a clear JPEG or PNG image."

    transform = transforms.Compose([
        transforms.Resize((bundle["img_size"], bundle["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(bundle["normalize_mean"], bundle["normalize_std"]),
    ])
    with torch.no_grad():
        probabilities = torch.softmax(bundle["model"](transform(image).unsqueeze(0)), dim=1)[0]
    top = int(probabilities.argmax())
    confidence = float(probabilities[top]) * 100
    predicted = bundle["class_names"][top]
    logger.info("leaf_prediction class=%s confidence=%.0f", predicted, confidence)
    return f"Predicted class: {predicted} (confidence: {confidence:.0f}%)"


class ClassifyLeafImageTool(TeaTool):
    name: str = "ClassifyLeafImage"
    description: str = (
        "Identifies the disease or pest shown in the tea leaf photo the farmer attached, using a "
        "trained image classification model. Takes no input - it already has the photo. Returns the "
        "predicted class name and a confidence percentage."
    )

    def _run(self) -> str:
        self.note_use()
        if not self.turn.image_path:
            return "No photo was attached to this question."
        return classify_leaf_image(self.turn.image_path)
