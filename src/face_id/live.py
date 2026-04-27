from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from face_id.gallery import load_gallery
from face_id.matching import match_embedding, match_embedding_knn

CENTER_TOLERANCE = 0.20
ELLIPSE_W_RATIO = 0.30
ELLIPSE_H_RATIO = 0.50
DEFAULT_LIVE_THRESHOLD = 0.45
DEFAULT_STABLE_DURATION = 2.0
DEFAULT_DISPLAY_DURATION = 3.0

GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
CYAN = (255, 255, 0)
DARK_GRAY = (100, 100, 100)


def _largest_face(faces: list) -> object | None:
    if not faces:
        return None
    return max(
        faces,
        key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
    )


class LiveRecognizer:
    def __init__(
        self,
        gallery_path: str | Path = "data/gallery.npz",
        camera_id: int = 0,
        device: str = "gpu",
        model: str = "buffalo_s",
        det_size: int = 320,
        max_side: int = 1280,
        threshold: float = DEFAULT_LIVE_THRESHOLD,
        matching: str = "centroid",
        knn_k: int = 3,
        stable_duration: float = DEFAULT_STABLE_DURATION,
        display_duration: float = DEFAULT_DISPLAY_DURATION,
        on_recognized: Callable[[str, float, bool], None] | None = None,
    ) -> None:
        gallery_path = Path(gallery_path)
        (
            self.labels,
            self.centroids,
            self.sample_labels,
            self.sample_embeddings,
            self.meta,
        ) = load_gallery(gallery_path)
        self.camera_id = camera_id
        self.device = device
        self.model_name = model
        self.det_size = det_size
        self.max_side = max_side
        self.threshold = threshold
        self.matching_mode = matching
        self.knn_k = knn_k
        self.stable_duration = stable_duration
        self.display_duration = display_duration
        self._app: FaceAnalysis | None = None
        self._running = True
        self.on_recognized = on_recognized

        self.state = "IDLE"
        self.stable_start: float | None = None
        self.recognized_name: str = ""
        self.recognized_similarity: float = 0.0
        self.recognized_accepted: bool = False
        self.recognized_time: float | None = None

    def stop(self) -> None:
        self._running = False

    @property
    def app(self) -> FaceAnalysis:
        if self._app is None:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device == "gpu"
                else ["CPUExecutionProvider"]
            )
            ctx_id = 0 if self.device == "gpu" else -1
            self._app = FaceAnalysis(name=self.model_name, providers=providers)
            self._app.prepare(ctx_id=ctx_id, det_size=(self.det_size, self.det_size))
        return self._app

    def _is_centered(self, face_bbox: np.ndarray, frame_h: int, frame_w: int) -> bool:
        x1, y1, x2, y2 = face_bbox[:4]
        face_cx = (x1 + x2) / 2
        face_cy = (y1 + y2) / 2
        max_dx = frame_w * CENTER_TOLERANCE
        max_dy = frame_h * CENTER_TOLERANCE
        return (
            abs(face_cx - frame_w / 2) <= max_dx
            and abs(face_cy - frame_h / 2) <= max_dy
        )

    def _extract_embedding(self, face: object) -> np.ndarray:
        emb = getattr(face, "normed_embedding", None)
        if emb is not None:
            return np.asarray(emb, dtype=np.float32)
        raw = np.asarray(face.embedding, dtype=np.float32)
        norm = np.linalg.norm(raw)
        return raw / norm if norm != 0 else raw

    def _match(self, embedding: np.ndarray) -> dict:
        if self.matching_mode == "knn":
            return match_embedding_knn(
                embedding,
                self.sample_labels,
                self.sample_embeddings,
                k=self.knn_k,
                threshold=self.threshold,
            )
        return match_embedding(
            embedding,
            self.labels,
            self.centroids,
            self.threshold,
        )

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()

        center = (w // 2, h // 2)
        axes = (int(w * ELLIPSE_W_RATIO), int(h * ELLIPSE_H_RATIO))

        ellipse_color = DARK_GRAY
        status_text = ""
        status_color = GRAY

        if self.state == "IDLE":
            ellipse_color = DARK_GRAY
            status_text = "No face detected"
            status_color = GRAY
        elif self.state == "TRACKING":
            if self.stable_start is not None:
                elapsed = time.time() - self.stable_start
                remaining = max(0.0, self.stable_duration - elapsed)
                if remaining > 0:
                    ellipse_color = CYAN
                    status_text = f"Hold still... {remaining:.1f}s"
                    status_color = CYAN
                else:
                    ellipse_color = CYAN
                    status_text = "Recognizing..."
                    status_color = CYAN
            else:
                ellipse_color = ORANGE
                status_text = "Align face in guide"
                status_color = ORANGE
        elif self.state == "RECOGNIZED":
            if self.recognized_accepted:
                ellipse_color = GREEN
                status_text = (
                    f"{self.recognized_name} - {self.recognized_similarity:.1f}%"
                )
                status_color = GREEN
            else:
                ellipse_color = RED
                status_text = "Unknown person"
                status_color = RED

        cv2.ellipse(
            overlay, center, axes, 0, 0, 360, ellipse_color, 2, cv2.LINE_AA
        )

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(
            overlay,
            f"Gallery: {len(self.labels)} person(s)",
            (20, 40),
            font,
            0.6,
            WHITE,
            2,
            cv2.LINE_AA,
        )

        if status_text:
            text_size = cv2.getTextSize(status_text, font, 0.8, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(
                overlay,
                status_text,
                (text_x, h - 60),
                font,
                0.8,
                status_color,
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            overlay,
            "[Q] Quit",
            (20, h - 20),
            font,
            0.5,
            GRAY,
            1,
            cv2.LINE_AA,
        )

        return overlay

    def process_frame(self, frame: np.ndarray, key: int) -> tuple[np.ndarray, bool]:
        """Process one frame for tick-based driving from the main thread.
        Returns (display_frame, should_stop).
        """
        if not self._running or key in (ord("q"), ord("Q")):
            return self._draw_overlay(frame), True

        faces = self.app.get(frame)
        face = _largest_face(faces)
        now = time.time()
        h, w = frame.shape[:2]

        if face is None:
            self.state = "IDLE"
            self.stable_start = None
        elif not self._is_centered(face.bbox, h, w):
            self.state = "TRACKING"
            self.stable_start = None
        else:
            if self.stable_start is None:
                self.stable_start = now
            elapsed = now - self.stable_start
            if elapsed >= self.stable_duration and self.state != "RECOGNIZED":
                embedding = self._extract_embedding(face)
                result = self._match(embedding)
                if result["accepted"]:
                    self.recognized_name = str(result["label"])
                    self.recognized_similarity = result["percent"]
                    self.recognized_accepted = True
                else:
                    self.recognized_name = "Unknown"
                    self.recognized_similarity = result["percent"]
                    self.recognized_accepted = False
                self.recognized_time = now
                self.state = "RECOGNIZED"
                if self.on_recognized is not None:
                    self.on_recognized(
                        self.recognized_name,
                        self.recognized_similarity,
                        self.recognized_accepted,
                    )

        if self.state == "RECOGNIZED" and self.recognized_time is not None:
            if now - self.recognized_time >= self.display_duration:
                self.state = "TRACKING"
                self.stable_start = None

        return self._draw_overlay(frame), False

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")

        _ = self.app
        cv2.startWindowThread()

        print(f"Live recognition started")
        print(f"Gallery: {list(self.labels)}")
        print(f"Threshold: {self.threshold}")
        print(f"Matching: {self.matching_mode}")
        print("Press [Q] to quit.\n")

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    continue

                faces = self.app.get(frame)
                face = _largest_face(faces)
                now = time.time()
                h, w = frame.shape[:2]

                if face is None:
                    self.state = "IDLE"
                    self.stable_start = None
                elif not self._is_centered(face.bbox, h, w):
                    self.state = "TRACKING"
                    self.stable_start = None
                else:
                    if self.stable_start is None:
                        self.stable_start = now

                    elapsed = now - self.stable_start

                    if (
                        elapsed >= self.stable_duration
                        and self.state != "RECOGNIZED"
                    ):
                        embedding = self._extract_embedding(face)
                        result = self._match(embedding)

                        if result["accepted"]:
                            self.recognized_name = str(result["label"])
                            self.recognized_similarity = result["percent"]
                            self.recognized_accepted = True
                        else:
                            self.recognized_name = "Unknown"
                            self.recognized_similarity = result["percent"]
                            self.recognized_accepted = False

                        self.recognized_time = now
                        self.state = "RECOGNIZED"

                        if self.on_recognized is not None:
                            self.on_recognized(
                                self.recognized_name,
                                self.recognized_similarity,
                                self.recognized_accepted,
                            )

                if self.state == "RECOGNIZED" and self.recognized_time is not None:
                    if now - self.recognized_time >= self.display_duration:
                        self.state = "TRACKING"
                        self.stable_start = None

                display = self._draw_overlay(frame)
                cv2.imshow("Live Face Recognition", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q") or not self._running:
                    break

        finally:
            cap.release()
            print("\nLive recognition stopped.")
