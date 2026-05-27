import os
from pathlib import Path

BASE_DIR = Path(__file__).parent


class Config:
    UPLOAD_FOLDER: str = str(BASE_DIR / "uploads")
    RESULTS_FOLDER: str = str(BASE_DIR / "results")
    MODEL_PATH: str = str(BASE_DIR / "models" / "best.pt")

    CONFIDENCE_THRESHOLD: float = 0.50
    SAMPLE_RATE: int = 6
    MAX_VIDEO_DURATION: int = 600
    ALLOWED_EXTENSIONS: set = {"mp4", "avi", "mov", "mkv"}

    MAX_CONTENT_LENGTH: int = 500 * 1024 * 1024  # 500 MB
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "crashvision-dev-key")

    @staticmethod
    def allowed_file(filename: str) -> bool:
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
        )
