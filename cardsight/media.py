from collections import Counter
from pathlib import Path
from typing import TypedDict, cast
from uuid import uuid4

import cv2

from cardsight.detector import (
    CardDetector,
    DetectionPayload,
    ImageArray,
    serialize_detections,
)
from cardsight.errors import MediaProcessingError


class VideoSummary(TypedDict):
    processed_frames: int
    total_detections: int
    detections_by_card: dict[str, int]


def detect_image(
    detector: CardDetector,
    image: ImageArray | None,
    confidence: float,
    iou: float,
) -> tuple[ImageArray, list[DetectionPayload]]:
    if image is None:
        raise MediaProcessingError("Choose an image before running detection.")
    annotated_image, detections = detector.detect(image, confidence, iou)
    return annotated_image, serialize_detections(detections)


def open_video(video_path: Path) -> tuple[cv2.VideoCapture, float, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise MediaProcessingError(
            f"OpenCV could not open video '{video_path}'. Use an MP4, MOV, AVI, or WebM file."
        )

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise MediaProcessingError(
            f"Video '{video_path}' has invalid metadata: fps={fps}, width={width}, height={height}."
        )
    return capture, fps, width, height


def open_video_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    codec = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), codec, fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise MediaProcessingError(
            f"OpenCV could not create output video '{output_path}' with the mp4v codec."
        )
    return writer


def summarize_video_detections(
    frame_count: int,
    detection_count: int,
    class_counts: Counter[str],
) -> VideoSummary:
    return {
        "processed_frames": frame_count,
        "total_detections": detection_count,
        "detections_by_card": dict(class_counts.most_common()),
    }


def detect_video(
    detector: CardDetector,
    video_path_value: str | None,
    confidence: float,
    iou: float,
    output_directory: Path,
) -> tuple[str, VideoSummary]:
    if video_path_value is None:
        raise MediaProcessingError("Choose or record a video before running detection.")

    input_path = Path(video_path_value)
    if not input_path.is_file():
        raise MediaProcessingError(f"Input video does not exist at '{input_path}'.")

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"detected-{uuid4().hex}.mp4"
    capture, fps, width, height = open_video(input_path)
    writer = open_video_writer(output_path, fps, width, height)
    frame_count = 0
    detection_count = 0
    class_counts: Counter[str] = Counter()

    try:
        while True:
            success, bgr_frame = capture.read()
            if not success:
                break
            rgb_frame = cast(
                ImageArray,
                cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB),
            )
            annotated_rgb, detections = detector.detect(rgb_frame, confidence, iou)
            writer.write(cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
            frame_count += 1
            detection_count += len(detections)
            class_counts.update(detection.card for detection in detections)
    finally:
        capture.release()
        writer.release()

    if frame_count == 0:
        output_path.unlink(missing_ok=True)
        raise MediaProcessingError(f"No video frames could be decoded from '{input_path}'.")

    summary = summarize_video_detections(frame_count, detection_count, class_counts)
    return str(output_path), summary
