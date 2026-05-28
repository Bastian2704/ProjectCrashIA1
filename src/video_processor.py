import logging
import uuid
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

    def process(self, video_path: str, job_id: str) -> List[Detection]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        results_dir = Path(Config.RESULTS_FOLDER) / job_id
        results_dir.mkdir(parents=True, exist_ok=True)

        detections: List[Detection] = []
        frame_index = 0

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

            found = self.detector.detect_frame(frame, frame_id=frame_index, timestamp=timestamp)

            for det in found:
                img_filename = f"frame_{frame_index:06d}.jpg"
                img_path = results_dir / img_filename
                annotated = self._draw_bbox(frame.copy(), det)
                cv2.imwrite(str(img_path), annotated)
                det.image_path = f"{job_id}/{img_filename}"
                detections.append(det)

            frame_index += 1

        cap.release()
        logger.info(f"Video procesado. {len(detections)} detecciones encontradas.")
        return detections

    @staticmethod
    def _draw_bbox(frame: np.ndarray, det: Detection) -> np.ndarray:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{det.label} {det.confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame
