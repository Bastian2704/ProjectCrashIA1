import logging
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.detector import AccidentDetector
from src.models import Detection
from config import Config

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Itera sobre los frames de un video, llama al detector y guarda
    los frames con detecciones positivas como imágenes JPEG.
    """

    def __init__(self, detector: AccidentDetector | None = None):
        self.detector = detector or AccidentDetector()

    def process(self, video_path: str, job_id: str, subdir: str = "") -> dict:
        """
        Procesa el video y devuelve un dict con:
          - detections: List[Detection]
          - stats: métricas de runtime (fps, timing, confianzas)
        subdir: subdirectorio dentro de results/job_id/ para guardar frames.
                Útil en modo comparación (ej. "n" o "s").
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        results_base = Path(Config.RESULTS_FOLDER) / job_id
        results_dir = results_base / subdir if subdir else results_base
        results_dir.mkdir(parents=True, exist_ok=True)

        detections: List[Detection] = []
        frame_index = 0
        frames_processed = 0
        inference_times: List[float] = []
        confidences: List[float] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % Config.SAMPLE_RATE != 0:
                frame_index += 1
                continue

            timestamp = frame_index / fps
            if timestamp > Config.MAX_VIDEO_DURATION:
                logger.info("Límite de duración alcanzado, deteniendo procesamiento.")
                break

            t0 = time.perf_counter()
            found = self.detector.detect_frame(frame, frame_id=frame_index, timestamp=timestamp)
            inference_times.append(time.perf_counter() - t0)
            frames_processed += 1

            for det in found:
                img_filename = f"frame_{frame_index:06d}.jpg"
                img_path = results_dir / img_filename
                annotated = self._draw_bbox(frame.copy(), det)
                cv2.imwrite(str(img_path), annotated)
                det.image_path = f"{job_id}/{subdir}/{img_filename}" if subdir else f"{job_id}/{img_filename}"
                detections.append(det)
                confidences.append(det.confidence)

            frame_index += 1

        cap.release()

        total_time = sum(inference_times)
        fps_inference = frames_processed / total_time if total_time > 0 else 0.0

        stats = {
            "frames_processed": frames_processed,
            "total_time_s": round(total_time, 3),
            "fps_inference": round(fps_inference, 2),
            "ms_per_frame": round((total_time / frames_processed) * 1000, 2) if frames_processed else 0,
            "confidence_min": round(min(confidences), 4) if confidences else None,
            "confidence_avg": round(sum(confidences) / len(confidences), 4) if confidences else None,
            "confidence_max": round(max(confidences), 4) if confidences else None,
        }

        logger.info(f"Video procesado [{subdir or 'default'}]. {len(detections)} detecciones encontradas.")
        return {"detections": detections, "stats": stats}

    @staticmethod
    def _draw_bbox(frame: np.ndarray, det: Detection) -> np.ndarray:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{det.label} {det.confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame
