import os
import uuid

from flask import Flask, jsonify, render_template, request

from deepfake_detect import detect_deepfake

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "pretrained_model", "ffpp_c40.pth")
ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 120 * 1024 * 1024


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detect", methods=["POST"])
def api_detect():
    if "video" not in request.files:
        return jsonify({"success": False, "message": "No video was uploaded."}), 400

    file = request.files["video"]
    if not file or not file.filename:
        return jsonify({"success": False, "message": "Choose a video file first."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify(
            {
                "success": False,
                "message": "Use MP4, AVI, MOV, MKV, or WEBM.",
            }
        ), 400

    if not os.path.exists(MODEL_PATH):
        return jsonify(
            {
                "success": False,
                "message": "Model file is missing: pretrained_model/ffpp_c40.pth",
            }
        ), 500

    save_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    try:
        file.save(save_path)
        result = detect_deepfake(save_path, MODEL_PATH)
        result["filename"] = file.filename
        return jsonify(result)
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500
    finally:
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except OSError:
                pass


if __name__ == "__main__":
    print("Open http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
