# Developer Documentation

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Technology Stack](#2-technology-stack)
3. [Architecture & Threading Model](#3-architecture--threading-model)
4. [End-to-End Workflow](#4-end-to-end-workflow)
5. [Module Reference](#5-module-reference)
6. [Adjustable Values](#6-adjustable-values)
7. [MQTT Integration](#7-mqtt-integration)

---

## 1. System Overview

Face Recognition Classifier is a zero-training, real-time face recognition system. It uses a pretrained ArcFace model (via InsightFace) to extract 512-dimensional face embeddings and matches them against a pre-built gallery using cosine similarity. No model fine-tuning or GPU training is required — only the gallery build step needs a set of reference images.

The system has three entry points:
- **GUI app** (`face-id app`) — unified Tkinter window with embedded camera feed
- **Standalone CLI tools** (`face-id enroll`, `face-id live`, `face-id build`, etc.)
- **Python API** — importable classes for integration in other systems

---

## 2. Technology Stack

| Library | Version | Purpose |
|---|---|---|
| **InsightFace** | ≥ 0.7.3 | Pretrained ArcFace model (`buffalo_s`) for face detection and 512-D embedding extraction |
| **ONNX Runtime** | ≥ 1.17 | Inference backend for InsightFace ONNX models; GPU variant uses CUDA EP |
| **OpenCV** | ≥ 4.8 | Camera capture (`VideoCapture`), image preprocessing, frame annotation, image I/O |
| **NumPy** | ≥ 1.24 | Embedding storage, cosine similarity via matrix dot products, gallery `.npz` serialization |
| **Tkinter** | stdlib | GUI control panel — status display, buttons, embedded camera label |
| **paho-mqtt** | ≥ 2.0 | MQTT client for publishing recognition events (optional extra) |
| **rich** | ≥ 13.7 | Terminal tables and colored output for CLI commands |
| **tqdm** | ≥ 4.66 | Progress bars during gallery build and batch prediction |

### InsightFace Model: `buffalo_s`

Downloaded automatically on first run to `~/.insightface/models/buffalo_s/`.

| Model file | Role |
|---|---|
| `det_500m.onnx` | Face detector (RetinaFace-style) |
| `w600k_mbf.onnx` | ArcFace recognition → 512-D embedding |
| `1k3d68.onnx` | 3D landmark detection (68 points) |
| `2d106det.onnx` | 2D landmark detection (106 points) |
| `genderage.onnx` | Gender and age estimation |

---

## 3. Architecture & Threading Model

### GUI App Threading

```
Main Thread (Tkinter event loop)
│
├── model_thread  ─── loads InsightFace ONNX models in background
│                     → calls root.after(0, _open_camera) when done
│
└── detection_thread ─── reads frames from VideoCapture
                          runs face detection + matching
                          encodes display frame as PNG
                          → calls root.after(0, _update_label, bytes)
                          (thread-safe via Tkinter's after scheduler)
```

- `VideoCapture` is opened and released on the **main thread**.
- All `cv2.imshow` / Tkinter label updates happen on the **main thread**.
- Face detection (GPU-heavy) runs in `detection_thread` to avoid blocking the UI.
- `_frame_pending` flag prevents frame queue buildup when GPU is faster than display.

### CLI Tools Threading

Standalone CLI commands (`face-id live`, `face-id enroll`) run their camera loop directly on the calling thread (single-threaded). They use `cv2.imshow` and `cv2.waitKey` in the same loop.

---

## 4. End-to-End Workflow

### Phase 1 — Enrollment

```
Camera frame
    └─► FaceEnroller.run() / process_frame()
            ├── Quality checks (size, blur, brightness, confidence)
            ├── Center alignment check (face within 20% of frame center)
            └── Save: images/<name>/enroll_NNN.jpg
```

### Phase 2 — Gallery Build

```
images/<person>/
    ├── photo1.jpg
    └── photo2.jpg
         │
         ▼
FaceRecognizer.build_gallery()
    ├── For each image: detect largest face → extract ArcFace embedding
    ├── Quality filter (optional)
    ├── Average all embeddings per person → centroid (L2-normalized)
    └── Save: data/gallery.npz
             ├── labels          (N,)       person names
             ├── centroids       (N, 512)   average embedding per person
             ├── sample_labels   (M,)       label for each sample
             ├── sample_embeddings (M, 512) all individual embeddings
             └── metadata        (JSON)     version, hash, model info
```

### Phase 3 — Live Recognition

```
Camera frame
    └─► LiveRecognizer.process_frame()
            ├── Detect largest face
            ├── State machine:
            │       IDLE ──(face appears)──► TRACKING
            │       TRACKING ──(stable 2s)──► RECOGNIZED
            │       RECOGNIZED ──(3s elapsed)──► TRACKING
            ├── On RECOGNIZED: extract embedding → match against gallery
            │       Centroid mode: cosine_similarity = centroids @ embedding
            │       KNN mode: top-K samples → majority vote
            └── Callback: on_recognized(name, similarity, accepted)
                    └─► MQTT publish (if enabled)
                    └─► CommandExecutor.execute()
                    └─► UI status update
```

### Cosine Similarity

Embeddings are L2-normalized, so cosine similarity reduces to a dot product:

```
similarity = embedding · centroid   ∈ [-1.0, 1.0]
percent    = similarity × 100       displayed to user
accepted   = similarity >= threshold
```

---

## 5. Module Reference

### `api.py` — Core Python API

**Class: `FaceRecognizer`**

Central high-level API. Wraps InsightFace model loading, image I/O, gallery build, prediction, and calibration.

| Method | Description |
|---|---|
| `build_gallery(images_dir, output, ...)` | Scans class folders, extracts embeddings, saves `.npz` gallery |
| `predict(input_path, gallery_path, ...)` | Runs recognition on image(s), returns `list[PredictResult]` |
| `calibrate(gallery_path)` | Leave-one-out cross-validation to find optimal threshold |
| `load_image(path)` | Reads image, downscales if longer side > `max_side` |

**Dataclass: `PredictResult`**

Immutable result object with fields: `image`, `person`, `similarity`, `similarity_percent`, `accepted`, `all_scores`.

---

### `gallery.py` — Gallery Serialization

Handles `.npz` file lifecycle. All gallery data is stored in a single compressed NumPy archive.

| Function | Description |
|---|---|
| `save_gallery(...)` | Writes labels, centroids, samples, metadata to `.npz`; embeds SHA256 of centroids |
| `load_gallery(path)` | Validates format, version, hash; returns 5-tuple of arrays + metadata dict |
| `validate_gallery(data)` | Checks required keys, version string, and centroid hash integrity |
| `build_metadata(...)` | Constructs the metadata dict stored inside the gallery |

**Gallery format version:** `1.0`  
**Validation:** SHA256 of the centroids array is stored in metadata and checked on every load.

---

### `matching.py` — Similarity Matching

Pure NumPy, no model calls. Operates on pre-extracted embeddings.

| Function | Mode | Description |
|---|---|---|
| `match_embedding(emb, labels, centroids, threshold)` | Centroid | Single dot product per class, takes argmax |
| `match_embedding_knn(emb, sample_labels, sample_embeddings, k, threshold)` | KNN | Scores all samples, takes top-K, majority vote, averages winner scores |
| `match_all_scores(emb, labels, centroids)` | Verbose | Returns sorted similarity list for all classes |

---

### `quality.py` — Face Quality Checks

Run during enrollment and gallery build to reject poor-quality images.

| Check | Function | Default threshold |
|---|---|---|
| Face size | `check_face_size` | ≥ 25 px (build) / ≥ 80 px (enroll) |
| Detection confidence | `check_det_confidence` | ≥ 0.5 |
| Blur (Laplacian variance) | `check_blur` | ≥ 5.0 |
| Brightness | `check_brightness` | 30 – 225 |

`check_face_quality()` chains all four checks in the order above, stopping at the first failure.

---

### `enroll.py` — Camera Enrollment

**Class: `FaceEnroller`**

Two modes of operation:

| Method | Used by | Description |
|---|---|---|
| `run()` | CLI (`face-id enroll`) | Full blocking camera loop with `cv2.imshow` |
| `begin_session()` + `process_frame(frame, key)` | GUI app | Tick-based; returns `(display_frame, done, result)` per frame |

Enrollment saves images as `images/<name>/enroll_NNN.jpg`. New captures append to existing images if the person folder already exists.

**Overlay drawn on each frame:**
- Ellipse face guide (center, 30% width × 50% height)
- Status text and color (green = ready, orange = quality issue, red = no face)
- Progress bar (top-right)
- Capture counter

---

### `live.py` — Live Recognition

**Class: `LiveRecognizer`**

Two modes of operation:

| Method | Used by | Description |
|---|---|---|
| `run()` | CLI (`face-id live`) | Full blocking camera loop with `cv2.imshow` |
| `process_frame(frame, key)` | GUI app | Tick-based; returns `(display_frame, should_stop)` per frame |

**State machine:**

```
IDLE ──────────(face detected & centered)──────► TRACKING
TRACKING ──────(stable for stable_duration)────► RECOGNIZED  (runs prediction)
RECOGNIZED ────(display_duration elapsed)──────► TRACKING
Any state ─────(face lost or not centered)─────► IDLE / TRACKING
```

The `on_recognized(name, similarity, accepted)` callback fires once per RECOGNIZED transition.

---

### `app.py` — GUI Application

**Class: `AppController`**

Manages the full lifecycle of the Tkinter window, camera threads, and state transitions.

**Key internal state:**

| Field | Type | Description |
|---|---|---|
| `_cap` | `cv2.VideoCapture` | Opened on main thread, read in detection thread |
| `_mode` | `str` | `"none"` / `"live"` / `"enroll"` |
| `_stop_detection` | `threading.Event` | Signals detection thread to exit |
| `_frame_pending` | `bool` | Prevents frame queue buildup |
| `_model_thread` | `Thread` | Loads InsightFace models in background |
| `_detection_thread` | `Thread` | Reads frames, runs detection, encodes PNG |
| `_mqtt` | `MqttNotifier \| None` | Publishes events if MQTT is enabled |

**Frame display pipeline:**

```
detection_thread:
    cap.read() → process_frame() → cv2.imencode('.png') → root.after(0, _update_label, bytes)

main thread (_update_label):
    tk.PhotoImage(data=base64(bytes)) → label.configure(image=photo)
```

> Note: `cv2.imencode` automatically converts BGR → RGB when writing PNG. Do **not** apply `cvtColor(BGR2RGB)` before encoding.

---

### `mqtt_client.py` — MQTT Notifier

**Class: `MqttNotifier`**

Thin wrapper around `paho-mqtt`. Uses `loop_start()` which spawns an internal network thread — all `publish()` calls are thread-safe and can be called from the detection thread.

| Method | Description |
|---|---|
| `connect()` | Connects to broker, starts network loop; publishes `sensors/off` on successful connect |
| `publish_approved(name, similarity)` | Publishes to `<prefix>/approved` |
| `publish_rejected()` | Publishes to `<prefix>/rejected` |
| `disconnect()` | Stops loop, closes connection |

---

### `command_executor.py` — Trigger Hook

Currently a stub that prints a terminal message on each accepted recognition. Replace the `execute()` method body to trigger external actions (webhooks, GPIO, shell commands, etc.).

```python
class CommandExecutor:
    def execute(self, name: str, similarity: float) -> None:
        print(f"[TRIGGER] Recognized: {name} ({similarity:.1f}%)")
```

---

### `cli.py` — CLI Entry Point

Parses arguments with `argparse` and delegates to the appropriate class. Each subcommand maps to one function: `_build_gallery`, `_predict`, `_calibrate`, `_enroll`, `_live`, `_app`. Common model flags (`--device`, `--model`, `--det-size`, `--max-side`) are injected via `_add_common_model_args()`.

---

## 6. Adjustable Values

### Recognition Thresholds

| Constant | Location | Default | Effect |
|---|---|---|---|
| `DEFAULT_THRESHOLD` | `api.py:16` | `0.38` | Used by `face-id predict` and `calibrate` |
| `--threshold` (live/app) | CLI | `0.45` | Higher = stricter; recommended for live to reduce false accepts |

### Quality Filter Defaults

Defined as module-level constants in `quality.py`:

| Constant | Default | Override via |
|---|---|---|
| `MIN_FACE_SIZE` | `25` px | `--min-face-size` (build: 25, enroll: 80) |
| `MIN_DET_CONFIDENCE` | `0.5` | `--min-det-conf` |
| `MIN_LAPLACIAN_VAR` | `5.0` | Code only (`check_blur` parameter) |
| `MIN_BRIGHTNESS` | `30.0` | Code only (`check_brightness` parameter) |
| `MAX_BRIGHTNESS` | `225.0` | Code only (`check_brightness` parameter) |

### Enrollment Behavior

Defined in `enroll.py`:

| Constant | Default | Description |
|---|---|---|
| `CAPTURE_DELAY` | `1.0` s | Minimum time between automatic captures |
| `CENTER_TOLERANCE` | `0.20` | Max face center offset as fraction of frame dimension |
| `ELLIPSE_W_RATIO` | `0.30` | Ellipse guide width = 30% of frame width |
| `ELLIPSE_H_RATIO` | `0.50` | Ellipse guide height = 50% of frame height |

### Live Recognition Timing

Defined in `live.py`:

| Constant | Default | CLI flag | Description |
|---|---|---|---|
| `DEFAULT_STABLE_DURATION` | `2.0` s | `--stable-duration` | Face must be centered and stable for this long before prediction fires |
| `DEFAULT_DISPLAY_DURATION` | `3.0` s | `--display-duration` | How long the result is shown before resetting to TRACKING |

### Model & Detection

| Parameter | Default | Description |
|---|---|---|
| `--model` | `buffalo_s` | InsightFace model pack; change to `buffalo_l` for higher accuracy |
| `--det-size` | `320` | Detector input resolution; `640` improves accuracy at cost of speed |
| `--max-side` | `1280` | Input images are downscaled if longer side exceeds this value |

### Gallery Version

`GALLERY_VERSION = "1.0"` in `gallery.py`. Increment this string when the gallery schema changes to invalidate old files gracefully.

---

## 7. MQTT Integration

### Connection Flow

```
cli.py _app()
    └── MqttNotifier(broker, port, topic_prefix).connect()
            └── paho loop_start()  ← background network thread
                    └── on_connect callback
                            └── _publish("sensors", "off")   ← startup signal
```

### Recognition Event Flow

```
detection_thread
    └── LiveRecognizer.process_frame()
            └── on_recognized(name, similarity, accepted)   ← callback
                    ├── root.after(0, _update_recognition_status)   ← UI update (main thread)
                    ├── CommandExecutor.execute()
                    └── MqttNotifier:
                            accepted == True  →  publish_approved(name, similarity)
                            accepted == False →  publish_rejected()
```

### Topics

| Topic | Trigger | Payload |
|---|---|---|
| `<prefix>/sensors` | App startup (broker connected) | `"off"` |
| `<prefix>/approved` | Face matched with sufficient similarity | `{"person": "name", "similarity": 87.5}` |
| `<prefix>/rejected` | Face detected but similarity below threshold | `{"person": "unknown"}` |

Default prefix: `face-id`. Override with `--mqtt-topic`.

### Thread Safety

`MqttNotifier.publish_*()` methods are called from `detection_thread`. This is safe because paho's `loop_start()` runs its own internal network thread and `client.publish()` is thread-safe.

### Adding Custom Topics

To publish additional events, add methods to `MqttNotifier` following the same pattern:

```python
def publish_custom(self, payload: str) -> None:
    self._publish("custom", payload)
```

Then call `self._mqtt.publish_custom(...)` from `AppController._on_recognized()` or any other event point.

### Extending `CommandExecutor`

`CommandExecutor.execute()` is called on every accepted recognition alongside MQTT. Use it for local side effects that don't require a broker (GPIO, shell scripts, file logging):

```python
class CommandExecutor:
    def execute(self, name: str, similarity: float) -> None:
        # Example: write to log file
        with open("recognition.log", "a") as f:
            f.write(f"{name},{similarity:.2f}\n")
```
