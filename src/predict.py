"""Inference helper used by the Streamlit app and the tests."""
import numpy as np

from .preprocessing import preprocess_bytes, to_batch


def predict_bytes(model, class_names, data: bytes) -> dict:
    img = preprocess_bytes(data)
    probs = model.predict(to_batch(img), verbose=0)[0].astype(float)
    top = int(np.argmax(probs))
    return {
        "label": class_names[top], "confidence": float(probs[top]), "index": top,
        "probabilities": {n: float(p) for n, p in zip(class_names, probs)},
        "model_input": img,
    }
