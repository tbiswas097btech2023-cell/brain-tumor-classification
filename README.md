# 🧠 NeuroScan AI — Brain Tumor MRI Classification

A Streamlit web app that classifies brain MRI slices with a CNN (TensorFlow/Keras) and reports class probabilities,
confidence, and **genuine, reproducible evaluation metrics** measured on a held-out test split.

> ⚠️ **Educational project only.** Not a medical device; never use it for diagnosis or treatment decisions.

## Current scope

| | |
|---|---|
| **Task** | 4-class classification: Glioma, Meningioma, No Tumor, Pituitary Tumor |
| **Dataset** | 7200 images -> 7013 unique (187 byte-identical duplicates removed) |
| **Split** | stratified 4909 / 1052 / 1052 (train / val / test), seed 42 |
| **Retrain** | `python train.py --dataset <folder>` (class folders -> classes; add `--exclude notumor` to drop a class) |


## Results (held-out test set, n = 1052)

| Metric | Value |
|---|---|
| Accuracy | **95.6 %** (95 % CI 94 - 97 %) |
| Precision / Recall / F1 (macro) | 95.7 % / 95.7 % / 95.6 % |
| ROC-AUC | 0.997 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Glioma | 0.996 | 0.892 | 0.941 | 268 |
| Meningioma | 0.931 | 0.955 | 0.943 | 268 |
| No Tumor | 0.947 | 0.984 | 0.965 | 252 |
| Pituitary Tumor | 0.956 | 0.996 | 0.976 | 264 |

Confusion matrix (rows = actual, columns = predicted):

| | Glioma | Meningioma | No Tumor | Pituitary Tumor |
|---|---|---|---|---|
| **Glioma** | 239 | 16 | 6 | 7 |
| **Meningioma** | 1 | 256 | 8 | 3 |
| **No Tumor** | 0 | 2 | 248 | 2 |
| **Pituitary Tumor** | 0 | 1 | 0 | 263 |

Full numbers: `reports/metrics.json`; plots: `reports/confusion_matrix.png`, `reports/training_curves.png`. Results come from one training run on one random split; they are not clinically validated.

## Features
- Upload an MRI (or pick a held-out sample) → label, confidence badge, gauge, probability chart, preprocessed-input preview
- **Performance dashboard**: accuracy/precision/recall/F1/AUC, confusion matrix, classification report, ROC, training curves, dataset split
- **Model info**: layer table, parameters, training configuration, library versions
- One shared preprocessing function (`src/preprocessing.py`) for training **and** inference; model cached with `st.cache_resource`

## Project structure
```
app.py                 Streamlit UI
train.py               Train + evaluate + export model/reports/sample images
src/                   config, preprocessing, data, model, predict
models/                brain_tumor_model.keras, class_names.json
reports/               metrics.json, confusion_matrix.png, training_curves.png
sample_images/         a few images from the held-out test split (for demo)
tests/                 pytest checks (preprocessing, model output, metrics consistency)
.streamlit/config.toml dark theme + upload limit
legacy/                original Flask app and notebook (kept for reference only)
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # Python 3.11 or 3.12
streamlit run app.py                                    # opens http://localhost:8501
```

## Retrain / evaluate (optional)
Dataset: [Brain MRI Images for Brain Tumor Detection (Kaggle)](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection).
Place it as `dataset/brain_tumor_dataset/{yes,no}/` (the `dataset/` folder is git-ignored), then:
```bash
python train.py                      # or: python train.py --dataset path/to/multiclass_dataset
pytest -q
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository (the model file is ~40 MB, below GitHub's 100 MB limit).
2. Go to <https://share.streamlit.io> → **Create app** → select the repo, branch `main`, main file `app.py`.
3. Under **Advanced settings** choose **Python 3.12**, then **Deploy**. No secrets are needed.

## Limitations
Small single-source dataset of 2-D slices; no check that an upload is really a brain MRI; metrics are not clinically validated.

## Author
Mandeep Kharb — original project. Upgraded to a Streamlit app with a reproducible evaluation pipeline.
