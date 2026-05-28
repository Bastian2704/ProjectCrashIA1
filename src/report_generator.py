import json
from pathlib import Path
from typing import List

from src.models import Detection
from config import Config


class ReportGenerator:
    """Genera y persiste el reporte JSON de un job de análisis."""

    def save(self, job_id: str, detections: List[Detection], video_filename: str) -> dict:
        report = {
            "job_id": job_id,
            "video_filename": video_filename,
            "total_detections": len(detections),
            "detections": [d.to_dict() for d in detections],
        }

        output_path = Path(Config.RESULTS_FOLDER) / job_id / "report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        return report

    def load(self, job_id: str) -> dict | None:
        report_path = Path(Config.RESULTS_FOLDER) / job_id / "report.json"
        if not report_path.exists():
            return None
        return json.loads(report_path.read_text(encoding="utf-8"))
