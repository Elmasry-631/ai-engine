import os

import torch


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

INDEX_PATH = os.path.join(STORAGE_DIR, "vector.index")
LABELS_PATH = os.path.join(STORAGE_DIR, "labels.json")


def resolve_device() -> torch.device:
    """Return a safe runtime device.

    Some environments report CUDA as available while the installed PyTorch/CUDA
    build is incompatible with the physical GPU capability. We probe CUDA with a
    tiny tensor op and fallback to CPU if it fails.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        _ = torch.zeros(1, device="cuda") + 1
        return torch.device("cuda")
    except Exception as exc:
        print(f"⚠️ CUDA unavailable at runtime, falling back to CPU: {exc}")
        return torch.device("cpu")


DEVICE = resolve_device()

EMBEDDING_DIM = 2048
THRESHOLD = 120
