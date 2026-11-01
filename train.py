"""Train + genuinely evaluate the model on a held-out test split.

Usage:  python train.py [--dataset PATH] [--epochs 100] [--batch-size 16]
Dataset layout: <dataset>/<class_name>/*.jpg  (optional <dataset>/validation/<class_name>/*.jpg).
Binary (yes/no) and multiclass (glioma/meningioma/pituitary/notumor) folders are both supported.
"""
import argparse
import json
import math
import platform
import shutil
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from src.config import (CLASSES_PATH, DATASET_DIR, DISPLAY_NAMES, IMG_SIZE, METRICS_PATH, MODEL_PATH, REPORTS_DIR,
                        SAMPLE_DIR, SEED)
from src.data import load_splits
from src.model import build_model


def wilson_ci(correct, n, z=1.96):
    p = correct / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - m) / d, (c + m) / d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATASET_DIR))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--exclude", nargs="*", default=[], help="class folders to ignore, e.g. --exclude notumor")
    args = ap.parse_args()

    keras.utils.set_random_seed(SEED)
    REPORTS_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    raw_names, (x_tr, y_tr), (x_va, y_va), (x_te, y_te, p_te), info = load_splits(__import__("pathlib").Path(args.dataset), exclude=args.exclude)
    names = [DISPLAY_NAMES.get(n.lower(), n.replace("_", " ").title()) for n in raw_names]
    print("Classes:", names, "\nData:", json.dumps(info, indent=1))

    cw = compute_class_weight("balanced", classes=np.arange(len(names)), y=y_tr)
    model = build_model(len(names))
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
    hist = model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=args.epochs, batch_size=args.batch_size,
                     class_weight=dict(enumerate(cw)), callbacks=[es], verbose=2)

    # ---- genuine evaluation on the untouched test split ----
    probs = model.predict(x_te, verbose=0)
    y_pred = probs.argmax(1)
    acc = accuracy_score(y_te, y_pred)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_te, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_te, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_te, y_pred, labels=range(len(names)))
    report = classification_report(y_te, y_pred, target_names=names, output_dict=True, zero_division=0)
    print(classification_report(y_te, y_pred, target_names=names, zero_division=0))

    roc = None
    if len(names) == 2:
        fpr, tpr, _ = roc_curve(y_te, probs[:, 1])
        roc = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc_score(y_te, probs[:, 1]))}
    elif len(names) > 2:
        roc = {"auc_ovr_macro": float(roc_auc_score(y_te, probs, multi_class="ovr", average="macro"))}

    # ---- artefacts ----
    model.save(MODEL_PATH)
    CLASSES_PATH.write_text(json.dumps(names))
    arch = []
    for l in model.layers:
        arch.append({"layer": l.name, "type": type(l).__name__, "output_shape": str(tuple(l.output.shape[1:])),
                     "params": int(l.count_params())})
    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classes": names, "raw_class_folders": raw_names, "dataset": info,
        "training": {"epochs_run": len(hist.history["loss"]), "epochs_max": args.epochs,
                     "best_epoch": int(np.argmin(hist.history["val_loss"])) + 1, "batch_size": args.batch_size,
                     "optimizer": "Adam", "loss": "sparse_categorical_crossentropy", "early_stopping": "val_loss, patience 12",
                     "class_weights": {names[i]: float(w) for i, w in enumerate(cw)}, "seed": SEED,
                     "input_size": list(IMG_SIZE), "total_params": int(model.count_params())},
        "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()},
        "test": {"n": int(len(y_te)), "accuracy": float(acc), "accuracy_ci95": wilson_ci(int((y_pred == y_te).sum()), len(y_te)),
                 "precision_macro": float(p_m), "recall_macro": float(r_m), "f1_macro": float(f_m),
                 "precision_weighted": float(p_w), "recall_weighted": float(r_w), "f1_weighted": float(f_w),
                 "confusion_matrix": cm.tolist(), "classification_report": report, "roc": roc},
        "architecture": arch,
        "environment": {"python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__},
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    # ---- static plots (also embedded in README) ----
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="mako", xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion Matrix (test n={len(y_te)})")
    plt.tight_layout(); plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150); plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k in ("accuracy", "val_accuracy"): ax[0].plot(hist.history[k], label=k)
    for k in ("loss", "val_loss"): ax[1].plot(hist.history[k], label=k)
    ax[0].set_title("Accuracy"); ax[1].set_title("Loss"); [a.legend() or a.set_xlabel("Epoch") for a in ax]
    plt.tight_layout(); plt.savefig(REPORTS_DIR / "training_curves.png", dpi=150); plt.close()

    # ---- demo samples: a few images from the held-out TEST split only ----
    if SAMPLE_DIR.exists(): shutil.rmtree(SAMPLE_DIR)
    SAMPLE_DIR.mkdir()
    from pathlib import Path
    for ci, raw in enumerate(raw_names):
        for p in [q for q, y in zip(p_te, y_te) if y == ci][:3]:
            shutil.copy(p, SAMPLE_DIR / f"{raw}_{Path(p).name}")
    print(f"\nTEST accuracy={acc:.4f}  macro-F1={f_m:.4f}  n={len(y_te)}  -> saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
