import os
from typing import List, Tuple

import gradio as gr
import torch
from PIL import Image
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"


def load_classifier(model_path: str) -> Tuple[torch.nn.Module, List[str]]:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    num_classes = checkpoint["num_classes"]
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    return model, class_names


MODEL, CLASS_NAMES = load_classifier(MODEL_PATH)

TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)


def predict_image(img: Image.Image) -> str:
    if img is None:
        raise ValueError("No image provided.")

    rgb_image = img.convert("RGB")
    input_tensor = TRANSFORM(rgb_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(input_tensor)
        _, preds = torch.max(outputs, 1)

    return CLASS_NAMES[preds.item()]


iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Predicted Class"),
    title="Image Classifier",
    description="Upload an image and the model will predict its class.",
)

if __name__ == "__main__":
    iface.launch()
