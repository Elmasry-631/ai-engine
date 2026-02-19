import os
import json
import faiss
import numpy as np
import time
import sys
from models.backbone import FeatureExtractor
from utils.config import DATA_DIR, INDEX_PATH, LABELS_PATH, EMBEDDING_DIM

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def build_index():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    extractor = FeatureExtractor()
    embeddings = []
    labels = []

    # جمع كل الصور أولًا عشان نعرف العدد الكلي
    all_images = []
    for class_name in os.listdir(DATA_DIR):
        class_path = os.path.join(DATA_DIR, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                _, ext = os.path.splitext(img_name)
                if ext.lower() not in VALID_IMAGE_EXTENSIONS:
                    continue
                img_path = os.path.join(class_path, img_name)
                all_images.append((img_path, class_name))

    total_images = len(all_images)
    if total_images == 0:
        raise ValueError(
            f"No valid images found under: {DATA_DIR}. "
            f"Supported extensions: {sorted(VALID_IMAGE_EXTENSIONS)}"
        )

    start_time = time.time()

    for idx, (img_path, class_name) in enumerate(all_images, 1):
        emb = extractor.extract(img_path)
        embeddings.append(emb)
        labels.append(class_name)

        # حساب الوقت والتقدير
        elapsed = time.time() - start_time
        avg_per_image = elapsed / idx
        remaining = avg_per_image * (total_images - idx)
        elapsed_min, elapsed_sec = divmod(int(elapsed), 60)
        remaining_min, remaining_sec = divmod(int(remaining), 60)

        # Progress Bar
        bar_len = 30
        filled_len = int(bar_len * idx / total_images)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write(
            f"\r[{bar}] {idx}/{total_images} | "
            f"Elapsed: {elapsed_min}m{elapsed_sec}s | "
            f"Remaining: {remaining_min}m{remaining_sec}s"
        )
        sys.stdout.flush()

    print("\n✅ Done building embeddings.")

    embeddings = np.array(embeddings)
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)

    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f)

    print("Index built successfully.")

if __name__ == "__main__":
    build_index()
