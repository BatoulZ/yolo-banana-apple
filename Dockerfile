# Python (CPU) slim base
FROM python:3.10-slim

# System deps for OpenCV + git (torch.hub fetches YOLOv5 repo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Faster/cleaner logs
ENV PYTHONUNBUFFERED=1

# App dir
WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code (incl. models/best.pt and templates)
COPY . .

# Bake YOLOv5 repo into the image to avoid GitHub rate limits at runtime
RUN git clone --depth 1 https://github.com/ultralytics/yolov5.git /app/yolov5

# Ensure uploads dir exists at runtime
RUN mkdir -p static/uploads

# App settings
ENV PORT=8000
ENV YOLO_CONF=0.25
ENV YOLO_IOU=0.45

# Expose port
EXPOSE 8000

# Production server
CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "4", "-b", "0.0.0.0:8000", "app:app"]
