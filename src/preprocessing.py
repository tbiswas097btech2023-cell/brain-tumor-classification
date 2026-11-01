"""Single source of truth for image preprocessing (used by training AND inference)."""
from pathlib import Path

import cv2
import numpy as np

from .config import IMG_SIZE


def preprocess_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes -> RGB, resize to IMG_SIZE, scale to [0, 1]. Returns (H, W, 3) float32."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the file as an image.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img.astype("float32") / 255.0


def preprocess_path(path) -> np.ndarray:
    return preprocess_bytes(Path(path).read_bytes())


def to_batch(img: np.ndarray) -> np.ndarray:
    return np.expand_dims(img, axis=0)
