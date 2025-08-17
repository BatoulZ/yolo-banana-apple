import os
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, Response
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import pathlib

# ---------- Config ----------
UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path("models/best.pt")  # your trained weights
ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp"}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")

def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# Windows checkpoint path fix
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

# ---------- Lazy model loader (safe) ----------
model = None
load_error = None  # remember last load error to avoid repeated crashes

def get_model():
    """
    Load YOLOv5 once.
    - In Docker/Render: use /app/yolov5 (cloned in the image)
    - Locally on Windows: pull from GitHub (cached by torch.hub)
    Any exception is captured and returned by callers instead of crashing the app.
    """
    global model, load_error
    if model is not None:
        return model

    try:
        import torch
        in_docker = Path("/app/yolov5").exists() or os.environ.get("DOCKER_ENV") == "1"
        if in_docker:
            repo = "/app/yolov5"; source = "local"; force = False
        else:
            repo = "ultralytics/yolov5"; source = "github"; force = True  # local dev

        m = torch.hub.load(
            repo, 'custom',
            path=str(MODEL_PATH),
            source=source,
            trust_repo=True,
            force_reload=force
        )
        m.conf = float(os.environ.get("YOLO_CONF", 0.25))
        m.iou  = float(os.environ.get("YOLO_IOU", 0.45))
        m.to("cpu")

        model = m
        load_error = None
        return model
    except Exception as e:
        load_error = str(e)
        return None

# ---------- Routes ----------
@app.route("/warmup")
def warmup():
    m = get_model()
    if m is None:
        # return the error text so the server doesn't crash
        return Response(f"Model load failed: {load_error}", status=500, mimetype="text/plain")
    return "ok", 200

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    # quick server health check: model must be ready
    m = get_model()
    if m is None:
        flash(f"Model not ready: {load_error}")
        return redirect(url_for("index"))

    if "image" not in request.files:
        flash("No file part.")
        return redirect(url_for("index"))

    f = request.files["image"]
    if f.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed(f.filename):
        flash("Unsupported file type. Upload JPG/PNG/BMP.")
        return redirect(url_for("index"))

    # Save upload
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    fname = secure_filename(f"{ts}_{f.filename}")
    src_path = UPLOAD_DIR / fname
    f.save(src_path)

    # Inference
    try:
        results = m(str(src_path))
        annotated_bgr = results.render()[0]

        out_name = src_path.stem + "_pred.jpg"
        out_path = UPLOAD_DIR / out_name
        Image.fromarray(annotated_bgr[:, :, ::-1]).save(out_path, quality=95)  # BGR->RGB

        detections = []
        boxes = results.xyxy[0].cpu().numpy() if hasattr(results.xyxy[0], "cpu") else results.xyxy[0].numpy()
        names = results.names
        for x1, y1, x2, y2, conf, cls_id in boxes:
            detections.append({"label": names[int(cls_id)], "conf": round(float(conf), 3)})

        return render_template(
            "result.html",
            input_image=url_for("uploaded_file", filename=fname),
            output_image=url_for("uploaded_file", filename=out_name),
            detections=detections,
        )

    except Exception as e:
        flash(f"Inference failed: {e}")
        return redirect(url_for("index"))

@app.route("/static/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)

if __name__ == "__main__":
    # Use port 8000 to match Docker/Render
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
