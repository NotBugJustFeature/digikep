from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATASET_ROOT: Path = PROJECT_ROOT / "dataset"
DATASET_DOWNLOAD_DIR: Path = DATASET_ROOT / "playing_cards"
DATASET_CONFIG_PATH: Path = DATASET_ROOT / "cards.yaml"
MODELS_DIR: Path = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH: Path = MODELS_DIR / "best.pt"
RUNS_DIR: Path = PROJECT_ROOT / "runs"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

KAGGLE_DATASET_HANDLE: str = "andy8744/playing-cards-object-detection-dataset"
DEFAULT_BASE_MODEL: str = "yolo26n.pt"
DEFAULT_RUN_NAME: str = "playing_cards_yolo11n"
DEFAULT_CONFIDENCE: float = 0.45
DEFAULT_IOU: float = 0.5
DEFAULT_IMAGE_SIZE: int = 640
LIVE_STREAM_INTERVAL_SECONDS: float = 0.25
