import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from utils.config import DEVICE

class FeatureExtractor:
    def __init__(self):
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model = self.model.to(DEVICE)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def extract(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            embedding = self.model(image)

        return embedding.view(-1).cpu().numpy().astype("float32")
