import gradio as gr
import torch
from torchvision import transforms, models
from PIL import Image
import os

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"

# --- LOAD MODEL ---
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
num_classes = checkpoint['num_classes']
class_names = checkpoint['class_names']

# استخدم نفس الكود: ResNet18 Transfer Learning
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()

# --- TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- PREDICTION FUNCTION ---
def predict_image(img):
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    else:
        img = img.convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)

    return class_names[preds.item()]

# --- GRADIO INTERFACE ---
iface = gr.Interface(
    fn=predict_image,   
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Predicted Class"),
    title="Image Classifier",
    description="Upload an image and the model will predict its class."
)

if __name__ == "__main__":
    iface.launch()

