from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from face_id.quality import check_face_quality

CAPTURE_DELAY = 1.0
CENTER_TOLERANCE = 0.20
ELLIPSE_W_RATIO = 0.30
ELLIPSE_H_RATIO = 0.50
NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")

GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)


def validate_name(name: str) -> None:
    if not NAME_PATTERN.match(name):
        raise ValueError(
            f"Invalid name '{name}'. "
            "Use English letters, digits, underscores. Must start with a letter."
        )


def _largest_face(faces: list) -> object | None:
    if not faces:
        return None
    return max(
        faces,
        key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
    )


class FaceEnroller:
    def __init__(
        self,
        name: str,
        camera_id: int = 0,
        images_dir: str | Path = "images",
        target_count: int = 10,
        min_face_size: int = 80,
        min_det_conf: float = 0.5,
        capture_delay: float = CAPTURE_DELAY,
        det_size: int = 320,
        max_side: int = 1280,
        device: str = "gpu",
        model: str = "buffalo_s",
        on_complete: Callable[[dict], None] | None = None,
    ) -> None:
        validate_name(name)
        self.name = name
        self.camera_id = camera_id
        self.images_dir = Path(images_dir)
        self.target_count = target_count
        self.min_face_size = min_face_size
        self.min_det_conf = min_det_conf
        self.capture_delay = capture_delay
        self.det_size = det_size
        self.max_side = max_side
        self.device = device
        self.model_name = model
        self._app: FaceAnalysis | None = None
        self._running = True
        self.on_complete = on_complete

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

    def _person_dir(self) -> Path:
        return self.images_dir / self.name

    @staticmethod
    def _next_index(person_dir: Path) -> int:
        if not person_dir.exists():
            return 1
        indices = []
        for p in person_dir.glob("enroll_*.jpg"):
            try:
                indices.append(int(p.stem.split("_")[1]))
            except (IndexError, ValueError):
                pass
        return max(indices) + 1 if indices else 1

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

    def _evaluate(
        self, face: object | None, frame: np.ndarray
    ) -> tuple[str, tuple[int, int, int], bool]:
        if face is None:
            return "No face detected", RED, False

        h, w = frame.shape[:2]
        if not self._is_centered(face.bbox, h, w):
            return "Move face to center", ORANGE, False

        ok, reason = check_face_quality(
            face,
            frame,
            min_face_size=self.min_face_size,
            min_det_conf=self.min_det_conf,
        )
        if not ok:
            return reason, ORANGE, False

        return "Ready", GREEN, True

    def _draw_overlay(
        self,
        frame: np.ndarray,
        captured: int,
        status: str,
        status_color: tuple[int, int, int],
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()

        center = (w // 2, h // 2)
        axes = (int(w * ELLIPSE_W_RATIO), int(h * ELLIPSE_H_RATIO))
        cv2.ellipse(overlay, center, axes, 0, 0, 360, (100, 100, 100), 2, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, f"Name: {self.name}", (20, 40), font, 0.8, WHITE, 2, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"Captured: {captured}/{self.target_count}",
            (20, 75),
            font,
            0.8,
            WHITE,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(overlay, status, (20, 110), font, 0.65, status_color, 2, cv2.LINE_AA)

        progress = captured / self.target_count if self.target_count > 0 else 0
        bar_w, bar_h = 200, 15
        bar_x = w - bar_w - 20
        bar_y = 30
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        cv2.rectangle(
            overlay,
            (bar_x, bar_y),
            (bar_x + int(bar_w * progress), bar_y + bar_h),
            GREEN,
            -1,
        )

        cv2.putText(
            overlay,
            "[Q] Quit   [SPACE] Force capture",
            (20, h - 20),
            font,
            0.5,
            GRAY,
            1,
            cv2.LINE_AA,
        )

        return overlay

    def begin_session(self) -> None:
        """Initialize state for tick-based operation (call once before process_frame)."""
        person_dir = self._person_dir()
        if person_dir.exists():
            existing = [
                f for f in person_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            if existing:
                print(
                    f"Warning: '{self.name}' already has {len(existing)} image(s). "
                    "New images will be added."
                )
        person_dir.mkdir(parents=True, exist_ok=True)
        self._session_dir = person_dir
        self._session_captured = 0
        self._session_last_time = 0.0
        self._session_next_idx = self._next_index(person_dir)

    def _session_result(self) -> dict:
        return {
            "name": self.name,
            "captured": self._session_captured,
            "target": self.target_count,
            "output_dir": str(self._session_dir),
        }

    def process_frame(
        self, frame: np.ndarray, key: int
    ) -> tuple[np.ndarray, bool, dict | None]:
        """Process one frame. Returns (display_frame, done, result_if_done)."""
        if not self._running or key in (ord("q"), ord("Q")):
            result = self._session_result()
            print(f"\nDone: {self._session_captured} image(s) saved to {self._session_dir}")
            display = self._draw_overlay(frame, self._session_captured, "Stopped", GRAY)
            return display, True, result

        faces = self.app.get(frame)
        face = _largest_face(faces)
        status, status_color, can_save = self._evaluate(face, frame)

        now = time.time()
        should_save = False
        if can_save and (now - self._session_last_time) >= self.capture_delay:
            should_save = True
        elif key == ord(" ") and can_save:
            should_save = True

        if should_save:
            filename = f"enroll_{self._session_next_idx:03d}.jpg"
            filepath = self._session_dir / filename
            cv2.imwrite(str(filepath), frame)
            self._session_captured += 1
            self._session_next_idx += 1
            self._session_last_time = now
            status = f"Captured! ({self._session_captured}/{self.target_count})"
            status_color = GREEN

        display = self._draw_overlay(frame, self._session_captured, status, status_color)

        if self._session_captured >= self.target_count:
            result = self._session_result()
            print(f"\nDone: {self._session_captured} image(s) saved to {self._session_dir}")
            return display, True, result

        return display, False, None

    def run(self) -> dict:
        person_dir = self._person_dir()

        if person_dir.exists():
            existing = [
                f
                for f in person_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            if existing:
                print(
                    f"Warning: '{self.name}' already has {len(existing)} image(s). "
                    "New images will be added."
                )

        person_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")

        captured = 0
        last_capture_time = 0.0
        next_index = self._next_index(person_dir)

        _ = self.app
        cv2.startWindowThread()

        print(f"Enrolling '{self.name}' - target: {self.target_count} images")
        print("Press [Q] to quit, [SPACE] to force capture.\n")

        try:
            while self._running and captured < self.target_count:
                ret, frame = cap.read()
                if not ret:
                    continue

                faces = self.app.get(frame)
                face = _largest_face(faces)

                status, status_color, can_save = self._evaluate(face, frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == ord("Q"):
                    break

                should_save = False
                now = time.time()

                if can_save and (now - last_capture_time) >= self.capture_delay:
                    should_save = True
                elif key == ord(" ") and can_save:
                    should_save = True

                if should_save:
                    filename = f"enroll_{next_index:03d}.jpg"
                    filepath = person_dir / filename
                    cv2.imwrite(str(filepath), frame)
                    captured += 1
                    next_index += 1
                    last_capture_time = now
                    status = f"Captured! ({captured}/{self.target_count})"
                    status_color = GREEN

                display = self._draw_overlay(frame, captured, status, status_color)
                cv2.imshow("Face Enrollment", display)

        finally:
            cap.release()

        print(f"\nDone: {captured} image(s) saved to {person_dir}")
        result = {
            "name": self.name,
            "captured": captured,
            "target": self.target_count,
            "output_dir": str(person_dir),
        }
        if self.on_complete is not None:
            self.on_complete(result)
        return result
