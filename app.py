from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("soybean_model.h5", compile=False)

# ---------------- LOAD CLASS NAMES ----------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ---------------- HOME ROUTE ----------------
@app.route("/")
def home():
    return "Soybean Disease Detection App Running ✅"

# ---------------- PREDICT ROUTE ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]

        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions)) * 100

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ---------------- RENDER PORT FIX ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
