# 🍌🍎 YOLOv5 Banana/Apple Detection

This project implements a **YOLOv5 object detection model** to identify bananas and apples in images.  
The model was trained on a custom dataset, packaged into a **Flask web app**, containerized with **Docker**, and deployed on **Render**.

---

## 🚀 Project Workflow

1. **Model Training**
   - Annotated and augmented dataset of bananas/apples
   - Trained YOLOv5 (`best.pt` exported after training)

2. **Flask Web App**
   - Upload an image via browser
   - Runs detection with YOLOv5
   - Returns an annotated image + list of detections

3. **Dockerization**
   - App containerized with a `Dockerfile`
   - Dependencies managed via `requirements.txt`

4. **Deployment**
   - Source code pushed to GitHub
   - Connected and deployed on [Render](https://render.com)

---

## 🗂️ Project Structure

```
model-to-deploy/
│
├── app.py                # Flask app entry point
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker container definition
├── .dockerignore         # Ignore unnecessary files in container
├── models/
│   └── best.pt           # YOLOv5 trained weights
├── templates/
│   ├── index.html        # Upload page
│   └── result.html       # Results page
└── static/
    └── uploads/          # Stores input & output images
```

---

## ⚙️ Setup & Run Locally

1. Clone this repo:
   ```bash
   git clone https://github.com/BatoulZ/yolo-banana-apple.git
   cd yolo-banana-apple
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run Flask app:
   ```bash
   python app.py
   ```
   Open [http://localhost:8000](http://localhost:8000)

---

## 🐳 Run with Docker

1. Build the image:
   ```bash
   docker build -t yolo-flask .
   ```

2. Run the container:
   ```bash
   docker run --rm -p 8000:8000 yolo-flask
   ```

3. Open [http://localhost:8000](http://localhost:8000)

---

## 🌐 Deployment

- The app is deployed on **Render**
- Connected directly to this GitHub repo
- Auto-builds on new commits

---

## 🔮 Future Improvements
- Deploy with GPU for faster inference
- Extend dataset to more fruits
- Add an API endpoint for external use

---

## 👩‍💻 Author
Batoul Zaiter
