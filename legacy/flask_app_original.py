from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
import os
import ssl
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img

# Disable SSL verification for downloading pre-trained models, if needed
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "./model/brain_tumor_detection_model.h5"  # Update with the actual model path
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Expand dimensions for model input
    return img


@app.route('/')
def home():
    return render_template('index.html')
    
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join("uploads", filename)

    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    file.save(filepath)

    # Preprocess and predict
    img = preprocess_image(filepath)
    prediction = model.predict(img)[0][0]
    
    # Define threshold
    threshold = 0.5
    result = "Tumor Detected" if prediction >= threshold else "No Tumor"

    return jsonify({"prediction": result, "confidence": f"{float(prediction):.3f}"})



if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
