import argparse
import html
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import gradio as gr

from cardsight.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    DEFAULT_MODEL_PATH,
    LIVE_STREAM_INTERVAL_SECONDS,
    OUTPUTS_DIR,
)
from cardsight.detector import CardDetector, DetectionPayload, ImageArray
from cardsight.media import VideoSummary, detect_image, detect_video

APP_CSS: str = """
:root {
  --cs-bg: #0f0f0f;
  --cs-surface: #191919;
  --cs-border: #343434;
  --cs-text: #f5f5f5;
  --cs-muted: #b5b5b5;
  --cs-primary: #2f8550;
  --cs-primary-hover: #276d43;
  --cs-accent: #d69e2e;
}

.gradio-container {
  background: var(--cs-bg) !important;
  color: var(--cs-text) !important;
  font-family: "IBM Plex Sans", "Helvetica Neue", sans-serif !important;
  max-width: 1240px !important;
  margin: 0 auto !important;
}

.cs-header {
  border-bottom: 1px solid var(--cs-border);
  padding: 20px 0 18px;
  margin-bottom: 20px;
}

.cs-header h1 {
  font-size: 26px;
  line-height: 1.2;
  margin: 0 0 6px;
}

.cs-header p {
  color: var(--cs-muted);
  font-size: 15px;
  margin: 0;
}

.cs-model-path {
  color: var(--cs-muted);
  border: 1px solid var(--cs-border);
  border-radius: 8px;
  background: var(--cs-surface);
  padding: 10px 12px;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  overflow-wrap: anywhere;
}

.gradio-container .tabs {
  border: 0 !important;
}

.gradio-container button.primary {
  background: var(--cs-primary) !important;
  border: 1px solid var(--cs-primary) !important;
  border-radius: 8px !important;
  color: #fff !important;
  min-height: 44px;
  transition: background-color 160ms ease, border-color 160ms ease;
}

.gradio-container button.primary:hover {
  background: var(--cs-primary-hover) !important;
  border-color: var(--cs-primary-hover) !important;
}

.gradio-container button:focus-visible,
.gradio-container input:focus-visible,
.gradio-container textarea:focus-visible {
  outline: 3px solid var(--cs-accent) !important;
  outline-offset: 2px;
}

.gradio-container .block {
  border-color: var(--cs-border) !important;
  border-radius: 10px !important;
  box-shadow: none !important;
}

.cs-help {
  color: var(--cs-muted);
  font-size: 14px;
  line-height: 1.55;
}

@media (max-width: 640px) {
  .gradio-container { padding: 0 14px !important; }
  .cs-header { padding-top: 16px; }
  .cs-header h1 { font-size: 22px; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition: none !important; animation: none !important; }
}
"""

APP_THEME: gr.themes.Base = gr.themes.Base(
    primary_hue=gr.themes.colors.green,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.gray,
    radius_size=gr.themes.sizes.radius_sm,
).set(
    body_background_fill="#0f0f0f",
    body_background_fill_dark="#0f0f0f",
    body_text_color="#f5f5f5",
    body_text_color_dark="#f5f5f5",
    body_text_color_subdued="#b5b5b5",
    body_text_color_subdued_dark="#b5b5b5",
    background_fill_primary="#191919",
    background_fill_primary_dark="#191919",
    background_fill_secondary="#202020",
    background_fill_secondary_dark="#202020",
    block_background_fill="#191919",
    block_background_fill_dark="#191919",
    block_border_color="#343434",
    block_border_color_dark="#343434",
    block_shadow="none",
    block_shadow_dark="none",
    input_background_fill="#202020",
    input_background_fill_dark="#202020",
    input_border_color="#454545",
    input_border_color_dark="#454545",
    button_primary_background_fill="#2f8550",
    button_primary_background_fill_dark="#2f8550",
    button_primary_background_fill_hover="#276d43",
    button_primary_background_fill_hover_dark="#276d43",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_transform_hover="none",
    button_transform_active="none",
)


@lru_cache(maxsize=4)
def load_detector(model_path_value: str) -> CardDetector:
    return CardDetector(Path(model_path_value))


def build_image_handler(
    detector_loader: Callable[[str], CardDetector],
    model_path: Path,
) -> Callable[[ImageArray | None, float, float], tuple[ImageArray, list[DetectionPayload]]]:
    def process_image(
        image: ImageArray | None,
        confidence: float,
        iou: float,
    ) -> tuple[ImageArray, list[DetectionPayload]]:
        detector = detector_loader(str(model_path.resolve()))
        return detect_image(detector, image, confidence, iou)

    return process_image


def build_video_handler(
    detector_loader: Callable[[str], CardDetector],
    model_path: Path,
) -> Callable[[str | None, float, float], tuple[str, VideoSummary]]:
    def process_video(
        video_path: str | None,
        confidence: float,
        iou: float,
    ) -> tuple[str, VideoSummary]:
        detector = detector_loader(str(model_path.resolve()))
        return detect_video(detector, video_path, confidence, iou, OUTPUTS_DIR)

    return process_video


def add_threshold_controls() -> tuple[gr.Slider, gr.Slider]:
    with gr.Row():
        confidence = gr.Slider(
            minimum=0.05,
            maximum=0.95,
            value=DEFAULT_CONFIDENCE,
            step=0.05,
            label="Confidence threshold",
        )
        iou = gr.Slider(
            minimum=0.1,
            maximum=0.9,
            value=DEFAULT_IOU,
            step=0.05,
            label="Overlap threshold (IoU)",
        )
    return confidence, iou


def build_app(model_path: Path) -> gr.Blocks:
    image_handler = build_image_handler(load_detector, model_path)
    video_handler = build_video_handler(load_detector, model_path)

    with gr.Blocks(title="CardSight") as app:
        gr.HTML(
            "<header class='cs-header'><h1>CardSight</h1>"
            "<p>Playing-card detection for images, recorded video, and a live camera.</p>"
            "</header>"
        )
        gr.HTML(
            "<div class='cs-model-path'><strong>Model</strong><br>"
            f"{html.escape(str(model_path.resolve()))}</div>"
        )

        with gr.Tabs():
            with gr.Tab("Image"):
                confidence, iou = add_threshold_controls()
                with gr.Row(equal_height=True):
                    image_input = gr.Image(
                        type="numpy",
                        sources=["upload", "clipboard"],
                        label="Input image",
                    )
                    image_output = gr.Image(type="numpy", label="Detected cards")
                image_button = gr.Button("Run image detection", variant="primary")
                image_detections = gr.JSON(label="Detections")
                image_button.click(
                    fn=image_handler,
                    inputs=[image_input, confidence, iou],
                    outputs=[image_output, image_detections],
                )

            with gr.Tab("Video"):
                video_confidence, video_iou = add_threshold_controls()
                with gr.Row(equal_height=True):
                    video_input = gr.Video(
                        sources=["upload", "webcam"],
                        format="mp4",
                        label="Input video",
                    )
                    video_output = gr.Video(label="Detected cards")
                video_button = gr.Button("Run video detection", variant="primary")
                video_summary = gr.JSON(label="Run summary")
                video_button.click(
                    fn=video_handler,
                    inputs=[video_input, video_confidence, video_iou],
                    outputs=[video_output, video_summary],
                    show_progress="full",
                )

            with gr.Tab("Live camera"):
                camera_confidence, camera_iou = add_threshold_controls()
                gr.HTML(
                    "<p class='cs-help'>Allow camera access, then start streaming. "
                    "Frames are processed locally by the configured model.</p>"
                )
                with gr.Row(equal_height=True):
                    camera_input = gr.Image(
                        type="numpy",
                        sources=["webcam"],
                        streaming=True,
                        label="Camera",
                    )
                    camera_output = gr.Image(type="numpy", label="Live detections")
                camera_detections = gr.JSON(label="Current detections")
                camera_input.stream(
                    fn=image_handler,
                    inputs=[camera_input, camera_confidence, camera_iou],
                    outputs=[camera_output, camera_detections],
                    stream_every=LIVE_STREAM_INTERVAL_SECONDS,
                    concurrency_limit=1,
                )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the CardSight detection app.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    app = build_app(arguments.model)
    app.launch(
        server_name=arguments.host,
        server_port=arguments.port,
        share=arguments.share,
        show_error=True,
        allowed_paths=[str(OUTPUTS_DIR.resolve())],
        css=APP_CSS,
        theme=APP_THEME,
    )


if __name__ == "__main__":
    main()
