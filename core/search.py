import faiss
import json
import numpy as np
from utils.config import INDEX_PATH, LABELS_PATH, THRESHOLD
from models.backbone import FeatureExtractor

def search_image(image_path):
    extractor = FeatureExtractor()

    index = faiss.read_index(INDEX_PATH)

    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)

    emb = extractor.extract(image_path)
    emb = np.expand_dims(emb, axis=0)

    D, I = index.search(emb, 1)

    distance = D[0][0]
    nearest_label = labels[I[0][0]]

    if distance < THRESHOLD:
        return nearest_label, distance
    else:
        return "NEW_CLASS", distance
