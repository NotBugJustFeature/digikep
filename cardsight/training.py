import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

from cardsight.config import (
    DATASET_CONFIG_PATH,
    DEFAULT_BASE_MODEL,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL_PATH,
    DEFAULT_RUN_NAME,
    RUNS_DIR,
)
from cardsight.errors import DatasetConfigurationError


@dataclass(frozen=True)
class TrainingConfig:
    dataset_config: Path
    base_model: str
    epochs: int
    batch_size: int
    image_size: int
    workers: int
    device: str | None


def validate_training_config(config: TrainingConfig) -> None:
    if not config.dataset_config.is_file():
        raise DatasetConfigurationError(
            f"Dataset config was not found at '{config.dataset_config}'. Run "
            "'uv run download_dataset.py' before training."
        )
    if config.epochs <= 0:
        raise ValueError(f"Epochs must be positive, received {config.epochs}.")
    if config.batch_size <= 0:
        raise ValueError(f"Batch size must be positive, received {config.batch_size}.")
    if config.image_size <= 0:
        raise ValueError(f"Image size must be positive, received {config.image_size}.")
    if config.workers < 0:
        raise ValueError(f"Workers cannot be negative, received {config.workers}.")


def training_arguments(config: TrainingConfig) -> dict[str, object]:
    arguments: dict[str, object] = {
        "data": str(config.dataset_config.resolve()),
        "epochs": config.epochs,
        "batch": config.batch_size,
        "imgsz": config.image_size,
        "workers": config.workers,
        "project": str(RUNS_DIR.resolve()),
        "name": DEFAULT_RUN_NAME,
        "exist_ok": True,
        "pretrained": True,
        "plots": True,
        "seed": 42,
    }
    if config.device is not None:
        arguments["device"] = config.device
    return arguments


def train_detector(config: TrainingConfig) -> Path:
    validate_training_config(config)
    model = YOLO(config.base_model)
    model.train(**training_arguments(config))
    trainer_value: object = model.trainer
    save_directory_value: object = getattr(trainer_value, "save_dir", None)
    if not isinstance(save_directory_value, (str, Path)):
        raise TypeError(
            "Training finished without returning an Ultralytics save directory. "
            f"Received trainer type '{type(trainer_value).__name__}'."
        )
    save_directory = Path(save_directory_value)
    best_weights = save_directory / "weights" / "best.pt"
    if not best_weights.is_file():
        raise RuntimeError(
            f"Training finished without creating expected weights at '{best_weights}'. "
            f"Inspect the run output in '{save_directory}'."
        )

    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, DEFAULT_MODEL_PATH)
    return DEFAULT_MODEL_PATH.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the playing-card YOLO detector.")
    parser.add_argument("--data", type=Path, default=DATASET_CONFIG_PATH)
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device, for example '0', 'cpu', or 'mps'.",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    config = TrainingConfig(
        dataset_config=arguments.data,
        base_model=arguments.model,
        epochs=arguments.epochs,
        batch_size=arguments.batch,
        image_size=arguments.image_size,
        workers=arguments.workers,
        device=arguments.device,
    )
    weights_path = train_detector(config)
    print(f"Best model: {weights_path}")


if __name__ == "__main__":
    main()
