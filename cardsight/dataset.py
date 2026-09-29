import argparse
import logging
import time
from pathlib import Path

import kagglehub
import yaml
from kagglehub.exceptions import (
    BackendError,
    CredentialError,
    DataCorruptionError,
    KaggleApiHTTPError,
    KaggleEnvironmentError,
    NotFoundError,
    UnauthenticatedError,
    UserCancelledError,
)

from cardsight.config import (
    DATASET_CONFIG_PATH,
    DATASET_DOWNLOAD_DIR,
    KAGGLE_DATASET_HANDLE,
)
from cardsight.errors import DatasetConfigurationError, DatasetDownloadError

LOGGER: logging.Logger = logging.getLogger(__name__)
SPLIT_DIRECTORY_NAMES: dict[str, tuple[str, ...]] = {
    "train": ("train",),
    "val": ("valid", "val"),
    "test": ("test",),
}


def download_dataset(destination: Path, force_download: bool, attempts: int = 3) -> Path:
    """Download the configured Kaggle dataset directly into the project."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            downloaded_path = kagglehub.dataset_download(
                KAGGLE_DATASET_HANDLE,
                output_dir=str(destination),
                force_download=force_download,
            )
            return Path(downloaded_path).resolve()
        except (
            BackendError,
            CredentialError,
            DataCorruptionError,
            KaggleApiHTTPError,
            KaggleEnvironmentError,
            NotFoundError,
            OSError,
            UnauthenticatedError,
            UserCancelledError,
        ) as error:
            last_error = error
            LOGGER.warning(
                "Dataset download attempt failed",
                extra={
                    "attempt": attempt,
                    "attempts": attempts,
                    "dataset": KAGGLE_DATASET_HANDLE,
                    "destination": str(destination),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            if attempt < attempts:
                time.sleep(float(attempt))

    message = (
        f"Could not download Kaggle dataset '{KAGGLE_DATASET_HANDLE}' to "
        f"'{destination}' after {attempts} attempts. Check network access and Kaggle "
        "credentials, then retry."
    )
    raise DatasetDownloadError(message) from last_error


def find_source_config(dataset_directory: Path) -> Path:
    configs = sorted(dataset_directory.rglob("data.yaml"))
    if not configs:
        raise DatasetConfigurationError(
            f"No data.yaml was found under '{dataset_directory}'. The downloaded "
            "dataset is not in the expected YOLO format."
        )
    if len(configs) > 1:
        paths = ", ".join(str(path) for path in configs)
        raise DatasetConfigurationError(
            f"Multiple data.yaml files were found under '{dataset_directory}': {paths}. "
            "Keep a single YOLO dataset configuration and retry."
        )
    return configs[0]


def find_split_directory(dataset_directory: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        direct_candidate = dataset_directory / name
        if (direct_candidate / "images").is_dir() and (direct_candidate / "labels").is_dir():
            return direct_candidate.resolve()

    for name in candidates:
        for nested_candidate in sorted(dataset_directory.rglob(name)):
            if (nested_candidate / "images").is_dir() and (nested_candidate / "labels").is_dir():
                return nested_candidate.resolve()

    return None


def extract_class_names(source_config: Path) -> list[str] | dict[int, str]:
    payload: object = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    names: object = payload.get("names") if isinstance(payload, dict) else None
    if isinstance(names, list) and all(isinstance(name, str) for name in names):
        return names
    if isinstance(names, dict) and all(
        isinstance(key, int) and isinstance(value, str) for key, value in names.items()
    ):
        return names
    raise DatasetConfigurationError(
        f"The dataset configuration '{source_config}' has no valid 'names' class mapping."
    )


def write_local_dataset_config(dataset_directory: Path, output_path: Path) -> Path:
    """Create a stable local YAML config instead of trusting downloaded relative paths."""
    source_config = find_source_config(dataset_directory)
    split_directories = {
        split: find_split_directory(dataset_directory, candidates)
        for split, candidates in SPLIT_DIRECTORY_NAMES.items()
    }
    train_directory = split_directories["train"]
    validation_directory = split_directories["val"]

    if train_directory is None or validation_directory is None:
        raise DatasetConfigurationError(
            f"Expected train/images, train/labels, valid/images, and valid/labels under "
            f"'{dataset_directory}'. Download the selected YOLO dataset again."
        )

    split_paths: dict[str, str] = {
        "train": str(train_directory / "images"),
        "val": str(validation_directory / "images"),
    }
    test_directory = split_directories["test"]
    if test_directory is not None:
        split_paths["test"] = str(test_directory / "images")

    config: dict[str, object] = {
        **split_paths,
        "names": extract_class_names(source_config),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_path.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and prepare the 52-class playing-card dataset."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing project-local dataset download.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = build_parser().parse_args()
    dataset_path = download_dataset(DATASET_DOWNLOAD_DIR, arguments.force)
    config_path = write_local_dataset_config(dataset_path, DATASET_CONFIG_PATH)
    print(f"Dataset: {dataset_path}")
    print(f"Training config: {config_path}")


if __name__ == "__main__":
    main()
