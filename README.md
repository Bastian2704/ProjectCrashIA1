# CrashVision

Automatic traffic accident detection from surveillance video. Upload a video, and CrashVision runs a fine-tuned YOLOv8 model frame by frame, highlights detected accidents with bounding boxes, and returns a timestamped gallery of events.

---

## Technologies

| Layer | Technology | Version |
|---|---|---|
| AI Model | Ultralytics YOLOv8 | ≥ 8.0 |
| Training | Google Colab + GPU T4 | — |
| Dataset | Roboflow Universe (YOLOv8 format) | 3 200+ images |
| Backend | Flask + Python | 3.10 |
| Video processing | OpenCV | 4.x |
| Frontend | HTML5 + Bootstrap + Vanilla JS | Bootstrap 5.3 |
| Storage | Local filesystem | — |

---

## Project Structure

```
ProjectCrashIA1/
├── app.py                  # Flask entry point + all routes
├── config.py               # Centralized configuration
├── requirements.txt
├── models/
│   └── best_n.pt           ← Download from Google Drive (see Installation)
├── src/
│   ├── detector.py         # AccidentDetector — YOLOv8 inference wrapper
│   ├── video_processor.py  # VideoProcessor — frame iteration + annotation
│   ├── report_generator.py # ReportGenerator — save/load JSON reports
│   └── models.py           # Detection dataclass
├── static/
│   ├── css/main.css
│   └── js/main.js
├── templates/
│   ├── base.html           # Bootstrap 5.3 dark layout
│   ├── index.html          # Upload form with progress bar
│   ├── results.html        # Detected frames gallery
│   └── 404.html
├── uploads/                # Uploaded videos (auto-created at runtime)
└── results/                # JPEG frames + JSON reports per job (auto-created)
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd ProjectCrashIA1
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install flask ultralytics opencv-python numpy Pillow python-dotenv
```

### 4. Place the trained model

Copy `best_n.pt` into the `models/` folder:

```
models/
└── best_n.pt
```

> **No model?** The app runs in **dummy mode** automatically — it generates random detections so you can test the full UI flow without a real model file.

---

## Running

```bash
python app.py
```

Open your browser at [http://localhost:5000](http://localhost:5000).

### Production (optional)

```bash
gunicorn -w 1 --timeout 300 -b 0.0.0.0:5000 app:app
```

> Use `-w 1` because the model is loaded into the main process memory.

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Upload page |
| `POST` | `/upload` | Receive video, process, return job result |
| `GET` | `/results/<job_id>` | HTML results page for a job |
| `GET` | `/api/results/<job_id>` | Full JSON report |
| `GET` | `/frames/<path>` | Serve annotated JPEG frames |

Accepted formats: `.mp4`, `.avi`, `.mov`, `.mkv` — max 500 MB, max 10 minutes.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `crashvision-dev-key` | Change in production to a random secure value |

---

## Authors

Sebastian Abad · Daniel Cornejo — UDLA, Inteligencia Artificial 1 (2026)
