from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO
from ultralytics.engine.results import Results

from cardsight.errors import ModelNotFoundError

ImageArray = NDArray[np.uint8]


@dataclass(frozen=True)
class Detection:
    card: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionPayload(TypedDict):
    card: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class CardDetector:
    """Typed connector around the Ultralytics inference interface."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise ModelNotFoundError(
                f"Model weights were not found at '{model_path}'. Run "
                "'uv run train.py' first, or pass --model with a trained .pt file."
            )
        self._model_path: Path = model_path.resolve()
        self._model: YOLO = YOLO(str(self._model_path))

    @property
    def model_path(self) -> Path:
        return self._model_path

    def detect(
        self,
        rgb_image: ImageArray,
        confidence: float,
        iou: float,
    ) -> tuple[ImageArray, list[Detection]]:
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        prediction_output: object = self._model.predict(
            source=bgr_image,
            conf=confidence,
            iou=iou,
            verbose=False,
            stream=False,
        )
        if not isinstance(prediction_output, list) or not prediction_output:
            raise TypeError(
                "Ultralytics inference returned no result list. "
                f"Received '{type(prediction_output).__name__}'."
            )
        result_value: object = prediction_output[0]
        if not isinstance(result_value, Results):
            raise TypeError(
                "Ultralytics inference returned an unexpected result type: "
                f"'{type(result_value).__name__}'."
            )
        result = result_value
        annotated_bgr: ImageArray = result.plot()
        annotated_rgb = cast(
            ImageArray,
            cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
        )
        return annotated_rgb, extract_detections(result)


def extract_detections(result: Results) -> list[Detection]:
    if result.boxes is None:
        return []

    names: dict[int, str] = result.names
    detections: list[Detection] = []
    for box in result.boxes:
        class_index = int(box.cls.item())
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        detections.append(
            Detection(
                card=names[class_index],
                confidence=round(float(box.conf.item()), 4),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    return detections


def serialize_detections(detections: list[Detection]) -> list[DetectionPayload]:
    return [cast(DetectionPayload, asdict(detection)) for detection in detections]
