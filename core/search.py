import faiss
import json
import os
import numpy as np
from utils.config import INDEX_PATH, LABELS_PATH, THRESHOLD
from models.backbone import FeatureExtractor

def search_image(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not os.path.isfile(INDEX_PATH):
        raise FileNotFoundError(f"Index file not found: {INDEX_PATH}")

    if not os.path.isfile(LABELS_PATH):
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

    extractor = FeatureExtractor()

    index = faiss.read_index(INDEX_PATH)

    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)

    emb = extractor.extract(image_path)
    emb = np.expand_dims(emb, axis=0)

    D, I = index.search(emb, 1)

    distance = float(D[0][0])
    nearest_idx = int(I[0][0])

    if nearest_idx < 0 or nearest_idx >= len(labels):
        raise IndexError(
            f"Predicted index {nearest_idx} out of bounds for labels length {len(labels)}"
        )

    nearest_label = labels[nearest_idx]

    if distance < THRESHOLD:
        return nearest_label, distance
    else:
        return "NEW_CLASS", distance
