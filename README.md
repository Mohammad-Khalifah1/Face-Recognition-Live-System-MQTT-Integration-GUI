# Face Recognition Classifier

Lightweight, zero-training face recognition using pretrained ArcFace embeddings. Enroll people from camera, build a gallery, and recognize faces in real-time — all locally with GPU support.

## Quick Start

```bash
uv sync --extra gpu          # or --extra cpu
uv run face-id app            # Launch GUI with live recognition
uv run face-id app --mqtt --mqtt-broker 192.168.1.100   # with MQTT
```

First run downloads the `buffalo_s` model (~125 MB) to `~/.insightface/models/`.

## Features

- **GUI application** — `face-id app` with Tkinter control panel + live OpenCV camera feed
- **Live recognition** — state machine (stable 2 s → predict once → display result)
- **Camera enrollment** — `face-id enroll` captures quality-filtered images from webcam
- **Image prediction** — `face-id predict` for batch or single image recognition
- **Threshold calibration** — leave-one-out cross-validation to find the optimal threshold
- **KNN matching** — optional K-nearest-neighbors mode for better discrimination
- **Quality filtering** — rejects blurry, small, dark, or low-confidence faces
- **Gallery integrity** — versioned format (v1.0) with SHA256 hash validation
- **GPU and CPU** — automatic CUDA fallback, configurable detection resolution
- **MQTT notifications** — publishes recognition events to any MQTT broker

## File Structure

```
face_recognition/
├── pyproject.toml
├── images/                    # Reference images (one folder per person)
│   ├── majd/
│   └── mohammad/
├── data/
│   └── gallery.npz            # Built gallery (gitignored)
├── tests/                     # Test images
├── src/face_id/
│   ├── api.py                 # FaceRecognizer — core Python API
│   ├── gallery.py             # Gallery build/load/save/validate
│   ├── matching.py            # Centroid + KNN matching
│   ├── quality.py             # Face quality checks
│   ├── enroll.py              # Camera enrollment (FaceEnroller)
│   ├── live.py                # Live recognition (LiveRecognizer)
│   ├── command_executor.py    # Trigger hook for recognized persons
│   ├── mqtt_client.py         # MQTT notifier (MqttNotifier)
│   ├── app.py                 # GUI application (Tkinter + OpenCV)
│   └── cli.py                 # CLI entry point
└── examples/
    └── predict_from_path.py
```

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra gpu                    # GPU (NVIDIA CUDA 12.x + cuDNN 9.x)
uv sync --extra cpu                    # CPU only
uv sync --extra gpu --extra mqtt       # GPU + MQTT
uv sync --extra cpu --extra mqtt       # CPU + MQTT
```

> **Note:** Always install `mqtt` together with `gpu` or `cpu`. Running `uv sync --extra mqtt` alone removes onnxruntime and breaks the app.

## CLI Commands

### `face-id app` — GUI Application

Launch the full application with live recognition, enrollment, and person management:

```bash
uv run face-id app
uv run face-id app --matching knn --threshold 0.45
uv run face-id app --mqtt --mqtt-broker 192.168.1.100
```

**Architecture:** Tkinter control panel for buttons and status + OpenCV camera feed embedded in the same window. Face detection runs in a background thread; only the label update happens on the main thread.

| Flag | Default | Description |
|---|---|---|
| `--gallery` | `data/gallery.npz` | Gallery file path |
| `--images-dir` | `images` | Reference images directory |
| `--camera` | `0` | Camera device ID |
| `--threshold` | `0.45` | Matching threshold |
| `--matching` | `centroid` | `centroid` or `knn` |
| `--knn-k` | `3` | KNN neighbors |
| `--device` | `gpu` | `gpu` or `cpu` |
| `--model` | `buffalo_s` | InsightFace model pack |
| `--det-size` | `320` | Face detector input size |
| `--max-side` | `1280` | Max image dimension before detection |
| `--mqtt` | off | Enable MQTT notifications |
| `--mqtt-broker` | `localhost` | MQTT broker IP address |
| `--mqtt-port` | `1883` | MQTT broker port |
| `--mqtt-topic` | `face-id` | MQTT topic prefix |

### `face-id enroll` — Enroll from Camera

Capture reference images from webcam with automatic quality filtering:

```bash
uv run face-id enroll --name mohammad --count 10
```

Opens a camera window with an ellipse face guide. Press **[SPACE]** to force capture, **[Q]** to quit.

| Flag | Default | Description |
|---|---|---|
| `--name` | (required) | Person name (English letters, digits, underscores) |
| `--count` | `10` | Number of images to capture |
| `--camera` | `0` | Camera device ID |
| `--images-dir` | `images` | Where to save captured images |
| `--min-face-size` | `80` | Minimum face size for enrollment |
| `--min-det-conf` | `0.5` | Minimum detection confidence |
| `--device` | `gpu` | `gpu` or `cpu` |
| `--model` | `buffalo_s` | InsightFace model pack |
| `--det-size` | `320` | Face detector input size |
| `--max-side` | `1280` | Max image dimension before detection |

### `face-id build` — Build Gallery

```bash
uv run face-id build --images-dir images --output data/gallery.npz
```

Detects faces, extracts embeddings, averages per person, saves gallery.

| Flag | Default | Description |
|---|---|---|
| `--images-dir` | `images` | Root directory of class folders |
| `--output` | `data/gallery.npz` | Gallery output path |
| `--skip-quality-check` | off | Disable quality filtering |
| `--min-face-size` | `25` | Minimum face size in pixels |
| `--min-det-conf` | `0.5` | Minimum detection confidence |
| `--device` | `gpu` | `gpu` or `cpu` |
| `--model` | `buffalo_s` | InsightFace model pack |
| `--det-size` | `320` | Face detector input size |
| `--max-side` | `1280` | Max image dimension before detection |

### `face-id predict` — Predict from Images

```bash
uv run face-id predict /path/to/photo.jpg --gallery data/gallery.npz
uv run face-id predict /path/to/folder --recursive --json
uv run face-id predict photo.jpg --matching knn --knn-k 5
```

| Flag | Default | Description |
|---|---|---|
| `input` | (required) | Image file or folder path |
| `--gallery` | `data/gallery.npz` | Gallery file path |
| `--threshold` | `0.38` | Cosine similarity threshold |
| `--matching` | `centroid` | `centroid` or `knn` |
| `--knn-k` | `3` | KNN neighbors |
| `--all-faces` | off | Match all faces in image |
| `--face-index` | (none) | Select Nth face (0-indexed) |
| `--recursive` | off | Scan subdirectories |
| `--verbose` | off | Show all match scores |
| `--json` | off | JSON output |
| `--device` | `gpu` | `gpu` or `cpu` |
| `--model` | `buffalo_s` | InsightFace model pack |
| `--det-size` | `320` | Face detector input size |
| `--max-side` | `1280` | Max image dimension before detection |

### `face-id live` — Live Recognition

Real-time recognition from camera. Predicts only after 2 s of face stability:

```bash
uv run face-id live --gallery data/gallery.npz
```

| Flag | Default | Description |
|---|---|---|
| `--gallery` | `data/gallery.npz` | Gallery file path |
| `--camera` | `0` | Camera device ID |
| `--threshold` | `0.45` | Matching threshold |
| `--matching` | `centroid` | `centroid` or `knn` |
| `--knn-k` | `3` | KNN neighbors |
| `--stable-duration` | `2.0` | Seconds of stability before predict |
| `--display-duration` | `3.0` | Seconds to display result |
| `--device` | `gpu` | `gpu` or `cpu` |
| `--model` | `buffalo_s` | InsightFace model pack |
| `--det-size` | `320` | Face detector input size |
| `--max-side` | `1280` | Max image dimension before detection |

### `face-id calibrate` — Calibrate Threshold

Find the optimal threshold via leave-one-out cross-validation (requires 2+ people):

```bash
uv run face-id calibrate --gallery data/gallery.npz
```

| Flag | Default | Description |
|---|---|---|
| `--gallery` | `data/gallery.npz` | Gallery file path |
| `--device` | `gpu` | `gpu` or `cpu` |
| `--model` | `buffalo_s` | InsightFace model pack |
| `--det-size` | `320` | Face detector input size |
| `--max-side` | `1280` | Max image dimension before detection |

## MQTT

Requires `paho-mqtt`: `uv sync --extra mqtt`

```bash
# Connect to broker at 192.168.1.100
uv run face-id app --mqtt --mqtt-broker 192.168.1.100

# Custom port and topic prefix
uv run face-id app --mqtt --mqtt-broker 192.168.1.100 --mqtt-port 1883 --mqtt-topic my-system
```

### Topics published

| Event | Topic | Payload |
|---|---|---|
| App startup | `face-id/sensors` | `"off"` |
| Person recognized (accepted) | `face-id/approved` | `{"person": "name", "similarity": 87.5}` |
| Face detected but not matched | `face-id/rejected` | `{"person": "unknown"}` |

The topic prefix (`face-id`) is configurable with `--mqtt-topic`.

### Python API (MQTT)

```python
from face_id.mqtt_client import MqttNotifier
from face_id.app import AppController

notifier = MqttNotifier(broker="192.168.1.100", port=1883, topic_prefix="face-id")
notifier.connect()

app = AppController(device="gpu", mqtt_notifier=notifier)
app.run()
```

## Python API

```python
from face_id.api import FaceRecognizer
from face_id.enroll import FaceEnroller
from face_id.live import LiveRecognizer
from face_id.app import AppController

# Build gallery
recognizer = FaceRecognizer(device="gpu")
result = recognizer.build_gallery(images_dir="images", output="data/gallery.npz")

# Predict
results = recognizer.predict(input_path="photo.jpg", gallery_path="data/gallery.npz")
for r in results:
    print(f"{r.person}: {r.similarity_percent:.2f}% (accepted={r.accepted})")

# Enroll from camera (standalone)
enroller = FaceEnroller(name="mohammad", target_count=10, device="gpu")
result = enroller.run()

# Live recognition (standalone)
live = LiveRecognizer(gallery_path="data/gallery.npz", threshold=0.45, matching="knn")
live.run()

# GUI application
app = AppController(device="gpu")
app.run()
```

## Threshold Guide

Similarity percentage is **cosine similarity × 100**, not a calibrated probability.

| Threshold | Behavior |
|---|---|
| 0.25 – 0.30 | Permissive — more false accepts |
| 0.32 – 0.38 | Balanced (build default: 0.38) |
| 0.40 – 0.50 | Strict — more false rejects (live default: 0.45) |
| 0.50+ | Very strict |

Run `face-id calibrate` to find the optimal threshold for your dataset.

## Matching Modes

- **Centroid** (default) — compares against the average embedding per person. Fast and simple.
- **KNN** — compares against all samples, takes top-K neighbors, votes on label. Better discrimination between similar-looking people.

## Quality Filtering

During `build` and `enroll`, images are checked automatically:

| Check | Threshold |
|---|---|
| Face size | ≥ 25 px (build) / ≥ 80 px (enroll) |
| Detection confidence | ≥ 0.5 |
| Blur (Laplacian variance) | ≥ 5.0 |
| Brightness | 30 – 225 |

## Tips for Better Accuracy

- Use 5–10 clear, frontal, well-lit photos per person
- Include variety in lighting and expressions
- Use `--det-size 640` for higher quality embeddings
- Use `--matching knn` for similar-looking people
- Run `face-id calibrate` after building the gallery

## GPU Notes

- Requires CUDA 12.x and cuDNN 9.x
- `--device gpu` falls back to CPU automatically if unavailable
- Minimum VRAM: ~500 MB for `buffalo_s`
- Use `--det-size 320 --max-side 1280` on low-VRAM GPUs (the defaults)
