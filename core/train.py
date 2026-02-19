import argparse
import copy
import os
from pathlib import Path
from typing import Dict, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "best_model.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image classifier and save best_model.pth")
    parser.add_argument("--data-dir", type=str, default=str(ROOT_DIR / "data"), help="Path to training data")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--output", type=str, default=str(MODEL_PATH), help="Checkpoint output path")
    return parser.parse_args()


def build_dataloaders(data_dir: str, batch_size: int, val_split: float) -> Tuple[object, object, Dict[int, str]]:
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    if len(dataset) == 0:
        raise ValueError(f"No training images found in: {data_dir}")

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Validation split is too high for current dataset size.")

    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    idx_to_class = {idx: name for name, idx in dataset.class_to_idx.items()}
    return train_loader, val_loader, idx_to_class


def train() -> None:
    args = parse_args()

    import torch
    from torch import nn, optim
    from torchvision import models

    from utils.config import DEVICE
    train_loader, val_loader, idx_to_class = build_dataloaders(
        args.data_dir, args.batch_size, args.val_split
    )

    num_classes = len(idx_to_class)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)

                preds = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = (correct / total) * 100 if total else 0.0

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": best_state,
        "num_classes": num_classes,
        "class_names": [idx_to_class[i] for i in range(num_classes)],
    }
    torch.save(checkpoint, output_path)
    print(f"✅ Saved best model to: {output_path}")
"""Gradio launch entrypoint."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import iface

"""Gradio launch entrypoint.

Note: this file previously duplicated app inference code. It now reuses
`app.iface` to avoid divergence and keep prediction behavior consistent.
"""

from app import iface


if __name__ == "__main__":
    train()
