import logging
import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from config import Config
from src.detector import AccidentDetector
from src.report_generator import ReportGenerator
from src.video_processor import VideoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Ensure runtime directories exist
Path(Config.UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(Config.RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)

detector = AccidentDetector()
detector.warmup()
processor = VideoProcessor(detector)
reporter = ReportGenerator()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """AI1-10: recibe el video, lanza análisis, devuelve job_id."""
    if "video" not in request.files:
        return jsonify({"error": "No se envió ningún archivo"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    if not Config.allowed_file(file.filename):
        return jsonify({"error": "Formato de archivo no permitido"}), 415

    job_id = str(uuid.uuid4())
    video_path = os.path.join(Config.UPLOAD_FOLDER, f"{job_id}_{file.filename}")
    file.save(video_path)

    try:
        detections = processor.process(video_path, job_id)
        report = reporter.save(job_id, detections, file.filename)
    except Exception as e:
        logger.error(f"Error procesando video {job_id}: {e}")
        return jsonify({"error": "Error interno al procesar el video"}), 500

    return jsonify({"job_id": job_id, "total_detections": report["total_detections"]}), 200


@app.route("/results/<job_id>")
def results_page(job_id: str):
    """AI1-11: página de resultados para un job."""
    report = reporter.load(job_id)
    if report is None:
        return render_template("404.html"), 404
    return render_template("results.html", report=report)


@app.route("/api/results/<job_id>")
def results_api(job_id: str):
    """Devuelve el reporte completo como JSON."""
    report = reporter.load(job_id)
    if report is None:
        return jsonify({"error": "Job no encontrado"}), 404
    return jsonify(report), 200


@app.route("/frames/<path:filename>")
def serve_frame(filename: str):
    """Sirve los frames guardados en results/."""
    return send_from_directory(Config.RESULTS_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
