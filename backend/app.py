from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import os, uuid, datetime
db=None
from backend.utils.preprocess import load_and_prepare
from backend.utils.gradcam import get_gradcam_heatmap, save_and_overlay_gradcam

from backend.config import IMAGE_SIZE, MODEL_PATH, CLASSES, GRADCAM_MIN_PROB

app = Flask(__name__)
CORS(app)

MODEL = tf.keras.models.load_model(MODEL_PATH)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

def save_file(fs):
    ext = os.path.splitext(fs.filename)[1].lower()
    fname = f"{uuid.uuid4()}{ext}"
    path = os.path.join(UPLOAD_DIR, fname)
    fs.save(path)
    return path

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    img_file = request.files["image"]
    if img_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    img_path = save_file(img_file)
    x = load_and_prepare(img_path)

    prob_uninfected = float(MODEL(x, training=False).numpy()[0][0])
    pred_idx = 1 if prob_uninfected >= 0.5 else 0
    label = CLASSES[pred_idx]
    predicted_probability = prob_uninfected if pred_idx == 1 else (1.0 - prob_uninfected)

    response = {
        "prediction": label,
        "predicted_probability": round(predicted_probability, 4)
    }

    if predicted_probability >= GRADCAM_MIN_PROB:
        heatmap = get_gradcam_heatmap(MODEL, x, target_class=pred_idx)
        out = os.path.join(UPLOAD_DIR, f"gradcam_{os.path.basename(img_path)}")
        save_and_overlay_gradcam(img_path, heatmap, out)
        response["gradcam_url"] = f"/uploads/{os.path.basename(out)}"
    if db:
        db.collection("history").add({
            "prediction": label,
            "predicted_probability": predicted_probability,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        })


    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
