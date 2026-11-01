"""Dataset discovery, de-duplication and stratified splitting.

Supported layouts (binary or multiclass, class names = folder names):
  dataset/<class>/*.jpg
  dataset/Training/<class>/*.jpg  +  dataset/Testing/<class>/*.jpg   (Kaggle 4-class layout)
All sub-folders are merged, de-duplicated, then re-split by train.py (so the test set is always unseen).
"""
import hashlib
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .config import DATASET_DIR, IMAGE_EXTS, IMG_SIZE, SEED
from .preprocessing import preprocess_path

SPLIT_DIRS = {"train", "training", "test", "testing", "val", "validation"}


def list_files(root: Path = DATASET_DIR, exclude=()):
    """Return (class_names, [(path, class_idx)]). `exclude` = class folder names to ignore (e.g. ['notumor'])."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    excl = {e.lower() for e in exclude}
    bases = [root] + [d for d in root.iterdir() if d.is_dir() and d.name.lower() in SPLIT_DIRS]
    names = sorted({d.name for b in bases for d in b.iterdir()
                    if d.is_dir() and d.name.lower() not in SPLIT_DIRS and d.name.lower() not in excl})
    if len(names) < 2:
        raise FileNotFoundError(f"Need at least 2 class folders inside {root}, found: {names}")
    items = []
    for idx, name in enumerate(names):
        for base in bases:
            folder = base / name
            if folder.is_dir():
                items += [(p, idx) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    return names, items


def dedupe(items):
    """Drop byte-identical files (duplicates would leak between train and test)."""
    seen, kept = set(), []
    for p, y in items:
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            kept.append((p, y))
    return kept


def load_splits(root: Path = DATASET_DIR, test_size=0.15, val_size=0.15, exclude=()):
    names, raw = list_files(root, exclude)
    items = dedupe(raw)
    paths = np.array([str(p) for p, _ in items])
    y = np.array([lbl for _, lbl in items])
    p_trainval, p_test, y_trainval, y_test = train_test_split(
        paths, y, test_size=test_size, stratify=y, random_state=SEED)
    p_train, p_val, y_train, y_val = train_test_split(
        p_trainval, y_trainval, test_size=val_size / (1 - test_size), stratify=y_trainval, random_state=SEED)

    def load(ps):  # pre-allocated to keep RAM low on large datasets
        out = np.empty((len(ps), IMG_SIZE[1], IMG_SIZE[0], 3), dtype="float32")
        for i, p in enumerate(ps):
            out[i] = preprocess_path(p)
        return out

    info = {
        "raw_images": len(raw), "unique_images": len(items), "duplicates_removed": len(raw) - len(items),
        "class_counts": {names[i]: int((y == i).sum()) for i in range(len(names))},
        "split_sizes": {"train": len(p_train), "val": len(p_val), "test": len(p_test)},
    }
    return names, (load(p_train), y_train), (load(p_val), y_val), (load(p_test), y_test, p_test), info
