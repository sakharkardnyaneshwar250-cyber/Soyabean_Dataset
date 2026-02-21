import os
import json
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Load model
model = load_model("soybean_model.h5", compile=False)

# Load class names
with open("class_names.json") as f:
    class_names = json.load(f)

# Load medicine data
with open("medicine_data.json", encoding="utf-8") as f:
    medicine_data = json.load(f)


def prepare_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["file"]
        image = Image.open(file.stream).convert("RGB")

        processed = prepare_image(image)
        prediction = model.predict(processed)

        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        disease = class_names[str(class_index)]

        info = medicine_data.get(disease, {})

        return render_template(
            "index.html",
            disease=disease,
            confidence=round(confidence, 2),
            info=info,
        )

    return render_template("index.html")


# IMPORTANT FOR RENDER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
