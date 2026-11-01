"""Run with: pytest -q   (requires a trained model: python train.py)"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES_PATH, IMG_SIZE, METRICS_PATH, MODEL_PATH, SAMPLE_DIR
from src.preprocessing import preprocess_bytes, preprocess_path

needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained")


def test_preprocessing_shape_and_range():
    img = preprocess_path(next(SAMPLE_DIR.glob("*")))
    assert img.shape == (IMG_SIZE[1], IMG_SIZE[0], 3) and img.dtype == np.float32
    assert 0.0 <= img.min() and img.max() <= 1.0


def test_bad_bytes_rejected():
    with pytest.raises(ValueError):
        preprocess_bytes(b"not an image")


@needs_model
def test_model_outputs_valid_distribution():
    from src.model import load_trained
    from src.predict import predict_bytes
    model, names = load_trained()
    assert model.output_shape[-1] == len(names)
    for f in SAMPLE_DIR.glob("*"):
        r = predict_bytes(model, names, f.read_bytes())
        assert abs(sum(r["probabilities"].values()) - 1) < 1e-4 and r["label"] in names


@needs_model
def test_metrics_report_is_self_consistent():
    m = json.loads(METRICS_PATH.read_text())["test"]
    cm = np.array(m["confusion_matrix"])
    assert cm.sum() == m["n"]
    assert abs(np.trace(cm) / cm.sum() - m["accuracy"]) < 1e-9
    assert json.loads(CLASSES_PATH.read_text()) == json.loads(METRICS_PATH.read_text())["classes"]
