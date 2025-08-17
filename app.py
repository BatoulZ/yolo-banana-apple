import os
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import numpy as np

# --- Config ---
UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path("models/best.pt")      # your YOLOv5-trained weights
ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp"}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")

def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# --- Load YOLOv5 model via torch.hub (NOT ultralytics v8) ---
# This pulls ultralytics/yolov5 the first time and caches it.

import os, pathlib
# Fix Linux-trained checkpoint on Windows
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath


model = torch.hub.load(
    '/app/yolov5',     # local path baked into the image
    'custom',
    path=str(MODEL_PATH),
    source='local',
    trust_repo=True
)

model.conf = 0.25
model.iou = 0.45
model.to('cpu')

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
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

    # --- Inference (YOLOv5) ---
    results = model(str(src_path))            # run prediction
    annotated_bgr = results.render()[0]       # annotated image (BGR)

    # Save annotated
    out_name = src_path.stem + "_pred.jpg"
    out_path = UPLOAD_DIR / out_name
    Image.fromarray(annotated_bgr[:, :, ::-1]).save(out_path, quality=95)  # BGR->RGB

    # Collect detections
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

@app.route("/static/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
