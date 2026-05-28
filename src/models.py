from dataclasses import dataclass
from typing import Tuple

@dataclass
class Detection:
    frame_id: int
    timestamp: float
    confidence: float
    bbox: Tuple[int, int, int, int]
    image_path: str = ""
    label: str = "accident"

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 3),
            "timestamp_str": self._format_timestamp(self.timestamp),
            "confidence": round(self.confidence, 4),
            "confidence_pct": f"{self.confidence * 100:.1f}%",
            "bbox": list(self.bbox),
            "image_path": self.image_path,
            "label": self.label,
        }

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        m, s = divmod(seconds, 60)
        return f"{int(m):02d}:{s:06.3f}"
