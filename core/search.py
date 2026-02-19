import json
import os
import numpy as np

from models.backbone import FeatureExtractor
from utils.config import INDEX_PATH, LABELS_PATH, THRESHOLD

def search_image(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not os.path.isfile(INDEX_PATH):
        raise FileNotFoundError(f"Index file not found: {INDEX_PATH}")

    if not os.path.isfile(LABELS_PATH):
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

    extractor = FeatureExtractor()

def search_image(image_path: str) -> Tuple[str, float]:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not os.path.isfile(INDEX_PATH):
        raise FileNotFoundError(f"Index file not found: {INDEX_PATH}")

    if not os.path.isfile(LABELS_PATH):
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

    extractor = FeatureExtractor()
    index = faiss.read_index(INDEX_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    if not labels:
        raise ValueError("Labels file is empty.")

    if index.ntotal != len(labels):
        raise ValueError(
            f"Index/labels size mismatch: index has {index.ntotal} vectors, "
            f"labels has {len(labels)} entries"
        )

    emb = extractor.extract(image_path)
    query = np.expand_dims(emb, axis=0).astype("float32")

    distances, indices = index.search(query, 1)
    distance = float(distances[0][0])
    nearest_idx = int(indices[0][0])

    distance = float(D[0][0])
    nearest_idx = int(I[0][0])

    if nearest_idx < 0 or nearest_idx >= len(labels):
        raise IndexError(
            f"Predicted index {nearest_idx} out of bounds for labels length {len(labels)}"
        )

    nearest_label = labels[nearest_idx]

    nearest_label = labels[nearest_idx]
    if distance < THRESHOLD:
        return nearest_label, distance
    return "NEW_CLASS", distance
