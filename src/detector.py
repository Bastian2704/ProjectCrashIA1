import logging
from pathlib import Path
from typing import List

import numpy as np

from src.models import Detection
from config import Config

logger = logging.getLogger(__name__)


class AccidentDetector:
    """
    Wrapper sobre el modelo YOLOv8 fine-tuned para detección de accidentes.
    Si el archivo .pt no existe, opera en modo dummy para desarrollo.
    """

    def __init__(
        self,
        model_path: str = Config.MODEL_PATH,
        confidence_threshold: float = Config.CONFIDENCE_THRESHOLD,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device: str = "cpu"
        self._dummy_mode: bool = False
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            logger.warning(
                f"Modelo no encontrado en {self.model_path}. "
                "Activando modo dummy para desarrollo."
            )
            self._dummy_mode = True
            return

        try:
            import torch
            from ultralytics import YOLO

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Cargando modelo desde {self.model_path} en {self.device}")
            self.model = YOLO(str(self.model_path))
            logger.info("Modelo cargado correctamente.")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}. Activando modo dummy.")
            self._dummy_mode = True

    def detect_frame(self, frame: np.ndarray, frame_id: int = 0, timestamp: float = 0.0) -> List[Detection]:
        if self._dummy_mode:
            return self._dummy_detect(frame_id, timestamp)

        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        detections: List[Detection] = []

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                    )
                )

        return detections

    def _dummy_detect(self, frame_id: int, timestamp: float) -> List[Detection]:
        """Retorna detecciones falsas cuando no hay modelo disponible."""
        import random
        if random.random() < 0.15:
            return [
                Detection(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    confidence=round(random.uniform(0.50, 0.95), 4),
                    bbox=(50, 50, 300, 250),
                )
            ]
        return []
