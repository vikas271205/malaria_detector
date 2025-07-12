from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import os, uuid
import datetime
from firebase_config import db
from utils.preprocess import load_and_prepare
from utils.gradcam import get_gradcam_heatmap, save_and_overlay_gradcam

app = Flask(__name__)
CORS(app)

MODEL_PATH = "malaria_model.h5"
MODEL = tf.keras.models.load_model(MODEL_PATH)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

def save_file(file_storage):
    ext = os.path.splitext(file_storage.filename)[1]
    fname = f"{uuid.uuid4()}{ext}"
    path = os.path.join(UPLOAD_DIR, fname)
    file_storage.save(path)
    return path

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    img_file = request.files["image"]
    if img_file.filename == "":
        return jsonify({"error": "Filename is empty."}), 400

    img_path = save_file(img_file)
    img_array = load_and_prepare(img_path)

    prob = MODEL.predict(img_array)[0][0]
    label = "Parasitized" if prob < 0.5 else "Uninfected"
    confidence = float(1 - prob) if prob < 0.5 else float(prob)

    # 🔥 Grad-CAM processing
    heatmap = get_gradcam_heatmap(MODEL, img_array, last_conv_layer_name="conv2d_2")
    gradcam_path = os.path.join(UPLOAD_DIR, f"gradcam_{os.path.basename(img_path)}")
    save_and_overlay_gradcam(img_path, heatmap, output_path=gradcam_path)

    db.collection("history").add({
        "prediction": label,
        "confidence": float(confidence * 100),
        "timestamp": datetime.datetime.utcnow().isoformat(),
    })

    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "gradcam_url": f"/uploads/{os.path.basename(gradcam_path)}"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
