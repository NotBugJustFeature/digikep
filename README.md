# CardSight

CardSight trains a YOLO model to recognize all 52 standard playing cards and runs the trained model on images, recorded video, or a live browser camera.

## What is included

- Project-local Kaggle dataset download to `dataset/playing_cards/`
- Normalized YOLO dataset configuration at `dataset/cards.yaml`
- YOLO11 training script with reproducible paths and settings
- Gradio application for image, video, and live camera detection
- Detection details for images/camera and aggregate counts for videos

The selected dataset is [Playing Cards Object Detection Dataset](https://www.kaggle.com/datasets/andy8744/playing-cards-object-detection-dataset). It contains synthetic playing-card scenes in YOLO format with 52 classes and train, validation, and test splits.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
```

## 1. Download the dataset

```bash
uv run download_dataset.py
```

The Kaggle files are downloaded directly to `dataset/playing_cards/`; they are not left in the global Kaggle cache. The command also writes `dataset/cards.yaml` with absolute local split paths. Public datasets normally download without authentication. If Kaggle asks for credentials, follow the [Kaggle API authentication instructions](https://github.com/Kaggle/kagglehub#authenticate).

To deliberately replace an existing download:

```bash
uv run download_dataset.py --force
```

## 2. Train the detector

The default command fine-tunes the small YOLO11 model for 50 epochs:

```bash
uv run train.py
```

Useful options:

```bash
uv run train.py --epochs 100 --batch 16 --image-size 640 --device 0
uv run train.py --epochs 50 --batch 8 --device mps
uv run train.py --epochs 5 --batch 4 --device cpu
```

Training outputs are written to `runs/playing_cards_yolo11n/`. The best weights are also copied to `models/best.pt`, which is the application default. GPU training is strongly recommended for the full dataset.

## 3. Run the application

```bash
uv run main.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860), then choose a workflow:

- **Image:** upload or paste an image, adjust thresholds, and run detection.
- **Video:** upload or record a clip and export an annotated MP4.
- **Live camera:** grant browser camera permission and stream detections in real time.

To use a different trained model:

```bash
uv run main.py --model /absolute/path/to/weights.pt
```

## Project layout

```text
cardsight/
  app.py          Gradio interface
  config.py       Project paths and defaults
  dataset.py      Kaggle download and YAML normalization
  detector.py     Ultralytics inference connector
  media.py        Image and video processing
  training.py     YOLO training pipeline
dataset/          Downloaded data (gitignored)
models/           Best trained weights (gitignored)
outputs/          Annotated videos (gitignored)
runs/             Ultralytics training runs (gitignored)
```

## Notes

- The training dataset is synthetic. For the best real-camera accuracy, add labeled photos from the actual table, lighting, card design, and camera angle you plan to use, then fine-tune the model.
- The app intentionally fails with a clear message if `models/best.pt` is missing; it does not silently substitute a generic model that cannot recognize card ranks and suits.
- Video inference processes every frame. Lowering the input resolution or using a GPU improves throughput.
