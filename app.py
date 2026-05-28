import json
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

# Default detector (YOLOv8n) pre-loaded for single-model mode
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
    """Recibe el video, lanza análisis en modo single o compare, devuelve job_id."""
    if "video" not in request.files:
        return jsonify({"error": "No se envió ningún archivo"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    if not Config.allowed_file(file.filename):
        return jsonify({"error": "Formato de archivo no permitido"}), 415

    mode = request.form.get("mode", "single")
    model_key = request.form.get("model", "n")

    job_id = str(uuid.uuid4())
    video_path = os.path.join(Config.UPLOAD_FOLDER, f"{job_id}_{file.filename}")
    file.save(video_path)

    try:
        if mode == "compare":
            return _process_compare(job_id, video_path, file.filename)
        else:
            return _process_single(job_id, video_path, file.filename, model_key)
    except Exception as e:
        logger.error(f"Error procesando video {job_id}: {e}", exc_info=True)
        return jsonify({"error": "Error interno al procesar el video"}), 500


def _process_single(job_id: str, video_path: str, filename: str, model_key: str):
    """Procesa con un solo modelo y guarda el report estándar."""
    if model_key == "s":
        det = AccidentDetector(model_path=Config.MODEL_S_PATH)
        proc = VideoProcessor(det)
    else:
        proc = processor  # reutiliza el detector n pre-cargado

    result = proc.process(video_path, job_id)
    report = reporter.save(job_id, result["detections"], filename)
    return jsonify({"job_id": job_id, "total_detections": report["total_detections"], "mode": "single"}), 200


def _process_compare(job_id: str, video_path: str, filename: str):
    """Procesa el mismo video con YOLOv8n y YOLOv8s y guarda un compare_report."""
    logger.info(f"[{job_id}] Iniciando modo comparación…")

    det_n = AccidentDetector(model_path=Config.MODEL_N_PATH)
    det_s = AccidentDetector(model_path=Config.MODEL_S_PATH)
    proc_n = VideoProcessor(det_n)
    proc_s = VideoProcessor(det_s)

    logger.info(f"[{job_id}] Procesando con YOLOv8n…")
    result_n = proc_n.process(video_path, job_id, subdir="n")

    logger.info(f"[{job_id}] Procesando con YOLOv8s…")
    result_s = proc_s.process(video_path, job_id, subdir="s")

    compare_report = {
        "job_id": job_id,
        "video_filename": filename,
        "mode": "compare",
        "models": {
            "n": {
                "total_detections": len(result_n["detections"]),
                "detections": [d.to_dict() for d in result_n["detections"]],
                "runtime_stats": result_n["stats"],
                "training_metrics": Config.TRAINING_METRICS["n"],
            },
            "s": {
                "total_detections": len(result_s["detections"]),
                "detections": [d.to_dict() for d in result_s["detections"]],
                "runtime_stats": result_s["stats"],
                "training_metrics": Config.TRAINING_METRICS["s"],
            },
        },
    }

    compare_path = Path(Config.RESULTS_FOLDER) / job_id / "compare_report.json"
    compare_path.parent.mkdir(parents=True, exist_ok=True)
    compare_path.write_text(json.dumps(compare_report, indent=2), encoding="utf-8")

    logger.info(f"[{job_id}] Comparación completada.")
    return jsonify({"job_id": job_id, "mode": "compare"}), 200


@app.route("/results/<job_id>")
def results_page(job_id: str):
    """Página de resultados para un job de modelo único."""
    report = reporter.load(job_id)
    if report is None:
        return render_template("404.html"), 404
    return render_template("results.html", report=report)


@app.route("/compare/<job_id>")
def compare_page(job_id: str):
    """Página de comparación lado a lado para un job de dos modelos."""
    compare_path = Path(Config.RESULTS_FOLDER) / job_id / "compare_report.json"
    if not compare_path.exists():
        return render_template("404.html"), 404
    report = json.loads(compare_path.read_text(encoding="utf-8"))
    return render_template("compare.html", report=report)


@app.route("/api/results/<job_id>")
def results_api(job_id: str):
    """Devuelve el reporte completo como JSON."""
    report = reporter.load(job_id)
    if report is None:
        return jsonify({"error": "Job no encontrado"}), 404
    return jsonify(report), 200


@app.route("/api/compare/<job_id>")
def compare_api(job_id: str):
    """Devuelve el compare report completo como JSON."""
    compare_path = Path(Config.RESULTS_FOLDER) / job_id / "compare_report.json"
    if not compare_path.exists():
        return jsonify({"error": "Job no encontrado"}), 404
    return jsonify(json.loads(compare_path.read_text(encoding="utf-8"))), 200


@app.route("/frames/<path:filename>")
def serve_frame(filename: str):
    """Sirve los frames guardados en results/."""
    return send_from_directory(Config.RESULTS_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
