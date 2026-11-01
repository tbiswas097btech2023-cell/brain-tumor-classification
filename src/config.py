"""Central configuration shared by training, evaluation and the Streamlit app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG_SIZE = (128, 128)                      # (width, height) - same as the original notebook
SEED = 42

MODEL_PATH = ROOT / "models" / "brain_tumor_model.keras"
CLASSES_PATH = ROOT / "models" / "class_names.json"
REPORTS_DIR = ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"
SAMPLE_DIR = ROOT / "sample_images"
DATASET_DIR = ROOT / "dataset" / "brain_tumor_dataset"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Folder name -> label shown in the UI. Unknown folder names are title-cased.
# Works unchanged if a multiclass dataset (glioma/meningioma/pituitary/notumor) is dropped in.
DISPLAY_NAMES = {
    "yes": "Tumor",
    "no": "No Tumor",
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary Tumor",
    "notumor": "No Tumor",
    "no_tumor": "No Tumor",
}
