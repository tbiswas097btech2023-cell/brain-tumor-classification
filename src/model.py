"""Model definition (original CNN backbone, upgraded to a softmax head + light augmentation)."""
import json

from tensorflow import keras
from tensorflow.keras import layers

from .config import CLASSES_PATH, IMG_SIZE, MODEL_PATH


def build_model(num_classes: int) -> keras.Model:
    """Same Conv(32/64/128) + Dense(128) design as the original notebook.
    Softmax output with `num_classes` units, so the identical code serves binary and multiclass data."""
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE[1], IMG_SIZE[0], 3)),
        layers.RandomFlip("horizontal"),           # augmentation: active only during training
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(), layers.Dropout(0.5),
        layers.Dense(128, activation="relu"), layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="brain_tumor_cnn")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def load_trained():
    """Load the saved model and its class names (call once and cache)."""
    model = keras.models.load_model(MODEL_PATH, compile=False)
    class_names = json.loads(CLASSES_PATH.read_text())
    return model, class_names
