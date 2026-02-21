import numpy as np
import json
import os
from flask import Flask, request, render_template, send_file
from tensorflow.keras.models import load_model
from PIL import Image
from gtts import gTTS

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

app = Flask(__name__)

model = load_model("soybean_model.h5", compile=False)

with open("class_names.json") as f:
    class_names = json.load(f)

with open("medicine_data.json", encoding="utf-8") as f:
    medicine_data = json.load(f)

def prepare_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Global variable for PDF storage
last_result = {}

@app.route("/", methods=["GET", "POST"])
def home():
    global last_result

    if request.method == "POST":
        file = request.files["file"]
        image = Image.open(file.stream).convert("RGB")

        processed = prepare_image(image)
        prediction = model.predict(processed)

        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100
        disease = class_names[str(class_index)]

        info = medicine_data.get(disease, {})

        # Save last result for PDF
        last_result = {
            "disease": disease,
            "confidence": round(confidence, 2),
            "medicine_en": info.get("medicine_en", ""),
            "medicine_mr": info.get("medicine_mr", ""),
            "dose_en": info.get("dose_en", ""),
            "dose_mr": info.get("dose_mr", ""),
            "advice_en": info.get("advice_en", ""),
            "advice_mr": info.get("advice_mr", "")
        }

        # Generate Marathi voice
        os.makedirs("static", exist_ok=True)
        tts = gTTS(text=f"तुमच्या पिकाला {disease} झाला आहे.", lang="mr")
        tts.save("static/voice.mp3")

        return render_template("index.html",
                               disease=disease,
                               confidence=round(confidence,2),
                               info=info,
                               voice=True)

    return render_template("index.html", voice=False)

@app.route("/download_pdf")
def download_pdf():
    global last_result

    if not last_result:
        return "No prediction available"

    file_path = "report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Soybean Disease Advisory Report", styles['Title']))
    elements.append(Spacer(1, 0.5 * inch))

    elements.append(Paragraph(f"Disease: {last_result['disease']}", styles['Normal']))
    elements.append(Paragraph(f"Confidence: {last_result['confidence']}%", styles['Normal']))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Medicine (English): {last_result['medicine_en']}", styles['Normal']))
    elements.append(Paragraph(f"Medicine (Marathi): {last_result['medicine_mr']}", styles['Normal']))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Dose (English): {last_result['dose_en']}", styles['Normal']))
    elements.append(Paragraph(f"Dose (Marathi): {last_result['dose_mr']}", styles['Normal']))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Advice (English): {last_result['advice_en']}", styles['Normal']))
    elements.append(Paragraph(f"Advice (Marathi): {last_result['advice_mr']}", styles['Normal']))

    doc.build(elements)

    return send_file(file_path, as_attachment=True)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
