from __future__ import annotations

import cv2
import numpy as np


MIN_FACE_SIZE = 25
MIN_DET_CONFIDENCE = 0.5
MIN_LAPLACIAN_VAR = 5.0
MIN_BRIGHTNESS = 30.0
MAX_BRIGHTNESS = 225.0


def check_blur(image: np.ndarray, min_var: float = MIN_LAPLACIAN_VAR) -> tuple[bool, str]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < min_var:
        return False, f"blurry (laplacian_var={lap_var:.1f} < {min_var})"
    return True, ""


def check_brightness(image: np.ndarray, min_b: float = MIN_BRIGHTNESS, max_b: float = MAX_BRIGHTNESS) -> tuple[bool, str]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    mean_val = float(np.mean(gray))
    if mean_val < min_b:
        return False, f"too dark (brightness={mean_val:.1f} < {min_b})"
    if mean_val > max_b:
        return False, f"too bright (brightness={mean_val:.1f} > {max_b})"
    return True, ""


def check_face_size(face_bbox: list | np.ndarray, min_size: int = MIN_FACE_SIZE) -> tuple[bool, str]:
    x1, y1, x2, y2 = face_bbox[:4]
    w, h = float(x2 - x1), float(y2 - y1)
    if w < min_size or h < min_size:
        return False, f"face too small ({w:.0f}x{h:.0f} < {min_size}x{min_size})"
    return True, ""


def check_det_confidence(face_det_score: float, min_conf: float = MIN_DET_CONFIDENCE) -> tuple[bool, str]:
    if face_det_score < min_conf:
        return False, f"low detection confidence ({face_det_score:.3f} < {min_conf})"
    return True, ""


def check_face_quality(
    face: object,
    image: np.ndarray,
    min_face_size: int = MIN_FACE_SIZE,
    min_det_conf: float = MIN_DET_CONFIDENCE,
    min_lap_var: float = MIN_LAPLACIAN_VAR,
    min_brightness: float = MIN_BRIGHTNESS,
    max_brightness: float = MAX_BRIGHTNESS,
) -> tuple[bool, str]:
    ok, reason = check_face_size(face.bbox, min_face_size)
    if not ok:
        return False, reason

    ok, reason = check_det_confidence(float(face.det_score), min_det_conf)
    if not ok:
        return False, reason

    x1, y1, x2, y2 = [int(v) for v in face.bbox[:4]]
    h_img, w_img = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    face_crop = image[y1:y2, x1:x2]
    if face_crop.size == 0:
        return False, "empty face crop"

    ok, reason = check_blur(face_crop, min_lap_var)
    if not ok:
        return False, reason

    ok, reason = check_brightness(face_crop, min_brightness, max_brightness)
    if not ok:
        return False, reason

    return True, ""
