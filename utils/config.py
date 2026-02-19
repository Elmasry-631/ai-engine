import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

INDEX_PATH = os.path.join(STORAGE_DIR, "vector.index")
LABELS_PATH = os.path.join(STORAGE_DIR, "labels.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBEDDING_DIM = 2048
THRESHOLD = 120 

