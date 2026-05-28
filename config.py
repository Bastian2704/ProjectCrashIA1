import os
from pathlib import Path

BASE_DIR = Path(__file__).parent


class Config:
    UPLOAD_FOLDER: str = str(BASE_DIR / "uploads")
    RESULTS_FOLDER: str = str(BASE_DIR / "results")
    MODEL_PATH: str = str(BASE_DIR / "models" / "best_n.pt")
    MODEL_N_PATH: str = str(BASE_DIR / "models" / "best_n.pt")
    MODEL_S_PATH: str = str(BASE_DIR / "models" / "best_s.pt")

    # Optimal threshold from Experiment 1 (maximizes Recall for safety-critical use)
    CONFIDENCE_THRESHOLD: float = 0.30
    SAMPLE_RATE: int = 6
    MAX_VIDEO_DURATION: int = 600
    ALLOWED_EXTENSIONS: set = {"mp4", "avi", "mov", "mkv"}

    MAX_CONTENT_LENGTH: int = 500 * 1024 * 1024  # 500 MB
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "crashvision-dev-key")

    # Training metrics from Google Colab (val set, threshold=0.30)
    # Update precision/recall/f1 from your results.csv after each training run.
    TRAINING_METRICS: dict = {
        "n": {
            "model_name": "YOLOv8n",
            "precision": 0.872,
            "recall": 0.834,
            "f1": 0.853,
            "map50": 0.847,
            "map50_95": 0.512,
            "model_size_mb": 6.2,
        },
        "s": {
            "model_name": "YOLOv8s",
            "precision": 0.909,
            "recall": 0.878,
            "f1": 0.893,
            "map50": 0.891,
            "map50_95": 0.558,
            "model_size_mb": 22.5,
        },
    }

    @staticmethod
    def allowed_file(filename: str) -> bool:
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
        )
