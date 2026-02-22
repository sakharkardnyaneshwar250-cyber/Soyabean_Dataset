from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json
import os

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
model = load_model("soybean_model.h5", compile=False)

# ---------------- LOAD CLASS NAMES ----------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ---------------- LOAD MEDICINE DATA ----------------
with open("medicine_data.json", "r") as f:
    medicine_data = json.load(f)


# ---------------- HOME ROUTE ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- PREDICTION ROUTE ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]

        # Open image
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))

        # Preprocess
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions)) * 100

        medicine_info = medicine_data.get(predicted_class, "No data available")

        return render_template(
            "result.html",
            prediction=predicted_class,
            confidence=round(confidence, 2),
            medicine=medicine_info
        )

    except Exception as e:
        return f"Error: {str(e)}"


# ---------------- PORT FIX FOR RENDER ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
