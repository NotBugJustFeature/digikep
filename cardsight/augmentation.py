import argparse
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import cv2
import numpy as np
import yaml
from numpy.random import Generator
from numpy.typing import NDArray

from cardsight.config import (
    AUGMENTATION_RANDOM_SEED,
    AUGMENTED_DATASET_CONFIG_PATH,
    AUGMENTED_DATASET_DIR,
    DATASET_CONFIG_PATH,
    DEFAULT_AUGMENTATION_VARIATIONS,
    DEFAULT_MAX_PERSPECTIVE_RATIO,
    DEFAULT_MAX_ROTATION_DEGREES,
    MIN_AUGMENTED_BOX_SIZE_PIXELS,
    MIN_AUGMENTED_BOX_VISIBILITY,
)
from cardsight.dataset import extract_class_names
from cardsight.errors import DatasetAugmentationError, DatasetConfigurationError

LOGGER: logging.Logger = logging.getLogger(__name__)
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})

ImageArray = NDArray[np.uint8]
TransformMatrix = NDArray[np.float64]


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass(frozen=True)
class DatasetPaths:
    train_images: Path
    train_labels: Path
    validation: str | list[str]
    test: str | list[str] | None


@dataclass(frozen=True)
class AugmentationConfig:
    dataset_config: Path
    output_directory: Path
    output_config: Path
    variations: int
    max_rotation_degrees: float
    max_perspective_ratio: float
    seed: int
    minimum_visibility: float
    minimum_box_size: float
    force: bool


class AugmentationSummary(TypedDict):
    source_images: int
    generated_images: int
    generated_boxes: int
    dropped_boxes: int
    dataset_config: str


def validate_augmentation_config(config: AugmentationConfig) -> None:
    if not config.dataset_config.is_file():
        raise DatasetConfigurationError(
            f"Dataset config was not found at '{config.dataset_config}'. Run "
            "'uv run download_dataset.py' before augmentation."
        )
    if config.variations <= 0:
        raise ValueError(f"Variations must be positive, received {config.variations}.")
    if not 0.0 <= config.max_rotation_degrees <= 180.0:
        raise ValueError(
            "Maximum rotation must be between 0 and 180 degrees, received "
            f"{config.max_rotation_degrees}."
        )
    if not 0.0 <= config.max_perspective_ratio < 0.5:
        raise ValueError(
            "Maximum perspective ratio must be at least 0 and below 0.5, received "
            f"{config.max_perspective_ratio}."
        )
    if not 0.0 <= config.minimum_visibility <= 1.0:
        raise ValueError(
            "Minimum box visibility must be between 0 and 1, received "
            f"{config.minimum_visibility}."
        )
    if config.minimum_box_size <= 0.0:
        raise ValueError(
            f"Minimum box size must be positive, received {config.minimum_box_size}."
        )


def validate_split_path(
    value: object,
    key: str,
    config_path: Path,
) -> str | list[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return cast(list[str], value)
    raise DatasetConfigurationError(
        f"Dataset config '{config_path}' has no valid '{key}' image path."
    )


def extract_required_split_path(
    payload: dict[str, object],
    key: str,
    config_path: Path,
) -> str | list[str]:
    value: object = payload.get(key)
    if value is None:
        raise DatasetConfigurationError(
            f"Dataset config '{config_path}' is missing required '{key}' image path."
        )
    return validate_split_path(value, key, config_path)


def extract_optional_split_path(
    payload: dict[str, object],
    key: str,
    config_path: Path,
) -> str | list[str] | None:
    value: object = payload.get(key)
    if value is None:
        return None
    return validate_split_path(value, key, config_path)


def resolve_config_path(value: str, config_path: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def resolve_split_path(
    value: str | list[str],
    config_path: Path,
) -> str | list[str]:
    if isinstance(value, str):
        return str(resolve_config_path(value, config_path))
    return [str(resolve_config_path(path, config_path)) for path in value]


def load_dataset_paths(config_path: Path) -> DatasetPaths:
    payload_value: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload_value, dict):
        raise DatasetConfigurationError(
            f"Dataset config '{config_path}' must contain a YAML mapping."
        )
    payload = cast(dict[str, object], payload_value)
    train_value = extract_required_split_path(payload, "train", config_path)
    if not isinstance(train_value, str):
        raise DatasetConfigurationError(
            f"Dataset config '{config_path}' must reference one original training image directory."
        )
    train_images = resolve_config_path(train_value, config_path)
    train_labels = train_images.parent / "labels"
    if not train_images.is_dir() or not train_labels.is_dir():
        raise DatasetConfigurationError(
            f"Expected training directories '{train_images}' and '{train_labels}'."
        )
    validation = resolve_split_path(
        extract_required_split_path(payload, "val", config_path),
        config_path,
    )
    test_value = extract_optional_split_path(payload, "test", config_path)
    test = resolve_split_path(test_value, config_path) if test_value is not None else None
    return DatasetPaths(
        train_images=train_images,
        train_labels=train_labels,
        validation=validation,
        test=test,
    )


def parse_yolo_box(line: str, line_number: int, label_path: Path) -> YoloBox:
    values = line.split()
    if len(values) != 5:
        raise DatasetAugmentationError(
            f"Invalid YOLO label at '{label_path}:{line_number}': expected 5 values, "
            f"received {len(values)}."
        )
    try:
        class_id = int(values[0])
        center_x, center_y, width, height = (float(value) for value in values[1:])
    except ValueError as error:
        raise DatasetAugmentationError(
            f"Invalid numeric YOLO label at '{label_path}:{line_number}': '{line}'."
        ) from error
    if class_id < 0:
        raise DatasetAugmentationError(
            f"Invalid negative class id at '{label_path}:{line_number}': {class_id}."
        )
    coordinates = {
        "center_x": center_x,
        "center_y": center_y,
        "width": width,
        "height": height,
    }
    invalid_coordinate = next(
        (name for name, value in coordinates.items() if not 0.0 <= value <= 1.0),
        None,
    )
    if invalid_coordinate is not None:
        raise DatasetAugmentationError(
            f"YOLO coordinate '{invalid_coordinate}' is outside [0, 1] at "
            f"'{label_path}:{line_number}'."
        )
    if width == 0.0 or height == 0.0:
        raise DatasetAugmentationError(
            f"YOLO box has zero area at '{label_path}:{line_number}'."
        )
    return YoloBox(class_id, center_x, center_y, width, height)


def read_yolo_labels(label_path: Path) -> list[YoloBox]:
    if not label_path.is_file():
        raise DatasetAugmentationError(
            f"Label file '{label_path}' is missing for its training image."
        )
    lines = label_path.read_text(encoding="utf-8").splitlines()
    return [
        parse_yolo_box(line, line_number, label_path)
        for line_number, line in enumerate(lines, start=1)
        if line.strip()
    ]


def create_transform_matrix(
    width: int,
    height: int,
    rng: Generator,
    max_rotation_degrees: float,
    max_perspective_ratio: float,
) -> TransformMatrix:
    angle = float(rng.uniform(-max_rotation_degrees, max_rotation_degrees))
    rotation = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    rotation_homography = np.vstack((rotation, np.array([0.0, 0.0, 1.0])))

    source_corners = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    jitter_scale = np.array(
        [width * max_perspective_ratio, height * max_perspective_ratio],
        dtype=np.float32,
    )
    corner_jitter = rng.uniform(-1.0, 1.0, size=(4, 2)).astype(np.float32) * jitter_scale
    destination_corners = source_corners + corner_jitter
    perspective = cv2.getPerspectiveTransform(source_corners, destination_corners)
    return cast(TransformMatrix, perspective @ rotation_homography)


def yolo_box_corners(box: YoloBox, width: int, height: int) -> NDArray[np.float32]:
    center_x = box.center_x * width
    center_y = box.center_y * height
    box_width = box.width * width
    box_height = box.height * height
    left = center_x - box_width / 2.0
    right = center_x + box_width / 2.0
    top = center_y - box_height / 2.0
    bottom = center_y + box_height / 2.0
    return np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]],
        dtype=np.float32,
    )


def transform_yolo_box(
    box: YoloBox,
    matrix: TransformMatrix,
    width: int,
    height: int,
    minimum_visibility: float,
    minimum_box_size: float,
) -> YoloBox | None:
    corners = yolo_box_corners(box, width, height).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    raw_left = float(np.min(transformed[:, 0]))
    raw_top = float(np.min(transformed[:, 1]))
    raw_right = float(np.max(transformed[:, 0]))
    raw_bottom = float(np.max(transformed[:, 1]))
    raw_width = max(0.0, raw_right - raw_left)
    raw_height = max(0.0, raw_bottom - raw_top)
    raw_area = raw_width * raw_height
    if raw_area == 0.0:
        return None

    left = min(max(raw_left, 0.0), float(width))
    top = min(max(raw_top, 0.0), float(height))
    right = min(max(raw_right, 0.0), float(width))
    bottom = min(max(raw_bottom, 0.0), float(height))
    clipped_width = max(0.0, right - left)
    clipped_height = max(0.0, bottom - top)
    visibility = (clipped_width * clipped_height) / raw_area
    if (
        visibility < minimum_visibility
        or clipped_width < minimum_box_size
        or clipped_height < minimum_box_size
    ):
        return None

    return YoloBox(
        class_id=box.class_id,
        center_x=((left + right) / 2.0) / width,
        center_y=((top + bottom) / 2.0) / height,
        width=clipped_width / width,
        height=clipped_height / height,
    )


def transform_boxes(
    boxes: list[YoloBox],
    matrix: TransformMatrix,
    width: int,
    height: int,
    minimum_visibility: float,
    minimum_box_size: float,
) -> tuple[list[YoloBox], int]:
    transformed = [
        transform_yolo_box(
            box,
            matrix,
            width,
            height,
            minimum_visibility,
            minimum_box_size,
        )
        for box in boxes
    ]
    visible_boxes = [box for box in transformed if box is not None]
    return visible_boxes, len(transformed) - len(visible_boxes)


def format_yolo_box(box: YoloBox) -> str:
    return (
        f"{box.class_id} {box.center_x:.6f} {box.center_y:.6f} "
        f"{box.width:.6f} {box.height:.6f}"
    )


def write_yolo_labels(label_path: Path, boxes: list[YoloBox]) -> None:
    content = "\n".join(format_yolo_box(box) for box in boxes)
    label_path.write_text(f"{content}\n" if content else "", encoding="utf-8")


def read_image(image_path: Path) -> ImageArray:
    image_value: object = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if not isinstance(image_value, np.ndarray) or image_value.ndim != 3:
        raise DatasetAugmentationError(
            f"OpenCV could not decode training image '{image_path}'."
        )
    return cast(ImageArray, image_value)


def write_image(image_path: Path, image: ImageArray) -> None:
    if not cv2.imwrite(str(image_path), image):
        raise DatasetAugmentationError(
            f"OpenCV could not write augmented image '{image_path}'."
        )


def list_training_images(images_directory: Path) -> list[Path]:
    images = sorted(
        path
        for path in images_directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise DatasetAugmentationError(
            f"No supported training images were found in '{images_directory}'."
        )
    return images


def augment_source_image(
    image_path: Path,
    label_path: Path,
    output_images: Path,
    output_labels: Path,
    config: AugmentationConfig,
    rng: Generator,
) -> tuple[int, int]:
    image = read_image(image_path)
    height, width = image.shape[:2]
    boxes = read_yolo_labels(label_path)
    generated_boxes = 0
    dropped_boxes = 0

    for variation in range(1, config.variations + 1):
        matrix = create_transform_matrix(
            width,
            height,
            rng,
            config.max_rotation_degrees,
            config.max_perspective_ratio,
        )
        transformed_image = cast(
            ImageArray,
            cv2.warpPerspective(
                image,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            ),
        )
        transformed_boxes, dropped = transform_boxes(
            boxes,
            matrix,
            width,
            height,
            config.minimum_visibility,
            config.minimum_box_size,
        )
        output_stem = f"{image_path.stem}__aug_{variation:02d}"
        write_image(output_images / f"{output_stem}{image_path.suffix.lower()}", transformed_image)
        write_yolo_labels(output_labels / f"{output_stem}.txt", transformed_boxes)
        generated_boxes += len(transformed_boxes)
        dropped_boxes += dropped

    return generated_boxes, dropped_boxes


def write_augmented_dataset_config(
    source_config: Path,
    paths: DatasetPaths,
    augmented_images: Path,
    output_config: Path,
) -> Path:
    config: dict[str, object] = {
        "train": [str(paths.train_images.resolve()), str(augmented_images.resolve())],
        "val": paths.validation,
        "names": extract_class_names(source_config),
    }
    if paths.test is not None:
        config["test"] = paths.test
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_config.resolve()


def augment_dataset(config: AugmentationConfig) -> AugmentationSummary:
    validate_augmentation_config(config)
    paths = load_dataset_paths(config.dataset_config)
    images = list_training_images(paths.train_images)
    output_exists = config.output_directory.exists() or config.output_config.exists()
    if output_exists and not config.force:
        raise FileExistsError(
            f"Augmented dataset already exists at '{config.output_directory}' or "
            f"'{config.output_config}'. Pass --force to replace generated data."
        )

    config.output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging_directory = Path(
        tempfile.mkdtemp(
            prefix=".cardsight-augmentation-",
            dir=config.output_directory.parent,
        )
    )
    output_images = staging_directory / "images"
    output_labels = staging_directory / "labels"
    output_images.mkdir()
    output_labels.mkdir()
    rng = np.random.default_rng(config.seed)
    generated_boxes = 0
    dropped_boxes = 0

    try:
        for image_index, image_path in enumerate(images, start=1):
            label_path = paths.train_labels / f"{image_path.stem}.txt"
            generated, dropped = augment_source_image(
                image_path,
                label_path,
                output_images,
                output_labels,
                config,
                rng,
            )
            generated_boxes += generated
            dropped_boxes += dropped
            if image_index % 500 == 0 or image_index == len(images):
                LOGGER.info(
                    "Dataset augmentation progress",
                    extra={"processed_images": image_index, "total_images": len(images)},
                )

        if config.output_directory.is_dir():
            shutil.rmtree(config.output_directory)
        elif config.output_directory.exists():
            raise DatasetAugmentationError(
                f"Augmented dataset output '{config.output_directory}' is not a directory."
            )
        staging_directory.replace(config.output_directory)
        augmented_config = write_augmented_dataset_config(
            config.dataset_config,
            paths,
            config.output_directory / "images",
            config.output_config,
        )
    finally:
        if staging_directory.exists():
            shutil.rmtree(staging_directory)

    return {
        "source_images": len(images),
        "generated_images": len(images) * config.variations,
        "generated_boxes": generated_boxes,
        "dropped_boxes": dropped_boxes,
        "dataset_config": str(augmented_config),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deterministic rotation and perspective augmentations for YOLO training data."
    )
    parser.add_argument("--data", type=Path, default=DATASET_CONFIG_PATH)
    parser.add_argument("--variations", type=int, default=DEFAULT_AUGMENTATION_VARIATIONS)
    parser.add_argument(
        "--max-rotation",
        type=float,
        default=DEFAULT_MAX_ROTATION_DEGREES,
        metavar="DEGREES",
    )
    parser.add_argument(
        "--max-perspective",
        type=float,
        default=DEFAULT_MAX_PERSPECTIVE_RATIO,
        metavar="RATIO",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing generated augmented dataset.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = build_parser().parse_args()
    config = AugmentationConfig(
        dataset_config=arguments.data,
        output_directory=AUGMENTED_DATASET_DIR,
        output_config=AUGMENTED_DATASET_CONFIG_PATH,
        variations=arguments.variations,
        max_rotation_degrees=arguments.max_rotation,
        max_perspective_ratio=arguments.max_perspective,
        seed=AUGMENTATION_RANDOM_SEED,
        minimum_visibility=MIN_AUGMENTED_BOX_VISIBILITY,
        minimum_box_size=MIN_AUGMENTED_BOX_SIZE_PIXELS,
        force=arguments.force,
    )
    summary = augment_dataset(config)
    print(f"Source images: {summary['source_images']}")
    print(f"Generated images: {summary['generated_images']}")
    print(f"Generated boxes: {summary['generated_boxes']}")
    print(f"Dropped boxes: {summary['dropped_boxes']}")
    print(f"Training config: {summary['dataset_config']}")


if __name__ == "__main__":
    main()
