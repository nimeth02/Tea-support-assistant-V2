"""ClassifyLeafImage tool: predicts the disease/pest shown in a tea leaf photo
using the ResNet18 transfer-learning model trained in ML Models/Tea Disease Classification/."""
import pickle
from pathlib import Path

import torch
from langchain.tools import Tool
from PIL import Image
from torchvision import transforms

import agent

MODEL_PATH = (
    Path(agent.__file__).resolve().parent.parent.parent
    / "ML Models" / "Tea Disease Classification" / "model" / "tea_disease_resnet18.pkl"
)

_bundle = None  # loaded once, reused for every call


def _load_bundle() -> dict:
    global _bundle
    if _bundle is None:
        with open(MODEL_PATH, "rb") as f:
            _bundle = pickle.load(f)
        _bundle["model"].eval()
    return _bundle


def classify_leaf_image(image_path: str) -> str:
    """Predict the disease/pest class for the tea leaf photo at image_path."""
    try:
        bundle = _load_bundle()
    except Exception as e:
        return f"Could not load the disease classification model: {e}"

    try:
        image = Image.open(image_path.strip()).convert("RGB")
    except Exception as e:
        return f"Could not open the image file ({image_path}): {e}"

    transform = transforms.Compose([
        transforms.Resize((bundle["img_size"], bundle["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(bundle["normalize_mean"], bundle["normalize_std"]),
    ])

    with torch.no_grad():
        output = bundle["model"](transform(image).unsqueeze(0))
        probabilities = torch.softmax(output, dim=1)[0]
        top_index = int(probabilities.argmax())
        confidence = float(probabilities[top_index]) * 100

    predicted_class = bundle["class_names"][top_index]
    return f"Predicted class: {predicted_class} (confidence: {confidence:.0f}%)"


leaf_image_classifier_tool = Tool(
    name="ClassifyLeafImage",
    func=classify_leaf_image,
    description=(
        "Identifies the disease or pest shown in a photo of a tea leaf using a trained image "
        "classification model. Input: the local file path to the uploaded image. Returns the "
        "predicted class name and a confidence percentage. Use this whenever the question "
        "mentions an attached image file path."
    ),
)
