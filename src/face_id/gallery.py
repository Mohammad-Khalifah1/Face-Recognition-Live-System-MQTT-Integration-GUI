from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

GALLERY_VERSION = "1.0"
REQUIRED_KEYS = {"labels", "centroids", "sample_labels", "sample_embeddings", "metadata"}


def gallery_hash(centroids: np.ndarray) -> str:
    return hashlib.sha256(centroids.tobytes()).hexdigest()[:16]


def build_metadata(
    images_dir: str,
    model: str,
    det_size: int,
    max_side: int,
    labels: list[str],
    skipped: dict[str, list[str]],
    threshold_recommendation: float | None = None,
) -> dict:
    return {
        "version": GALLERY_VERSION,
        "images_dir": str(images_dir),
        "model": model,
        "det_size": det_size,
        "max_side": max_side,
        "labels": labels,
        "skipped": skipped,
        "threshold_recommendation": threshold_recommendation,
    }


def save_gallery(
    path: Path,
    labels: list[str],
    centroids: list[np.ndarray],
    sample_labels: list[str],
    sample_embeddings: list[np.ndarray],
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    centroids_arr = np.vstack(centroids).astype(np.float32)
    metadata["centroids_sha256"] = hashlib.sha256(centroids_arr.tobytes()).hexdigest()
    np.savez_compressed(
        path,
        labels=np.asarray(labels),
        centroids=centroids_arr,
        sample_labels=np.asarray(sample_labels),
        sample_embeddings=np.vstack(sample_embeddings).astype(np.float32),
        metadata=json.dumps(metadata, ensure_ascii=False),
    )


def validate_gallery(data: np.lib.npyio.NpzFile) -> tuple[bool, str]:
    missing = REQUIRED_KEYS - set(data.files)
    if missing:
        return False, f"Missing keys in gallery: {missing}"

    meta_raw = str(data["metadata"])
    try:
        meta = json.loads(meta_raw)
    except (json.JSONDecodeError, TypeError):
        return False, "Gallery metadata is not valid JSON"

    version = meta.get("version", "0.0")
    if version != GALLERY_VERSION:
        return False, f"Gallery version mismatch: got {version}, expected {GALLERY_VERSION}"

    centroids = data["centroids"]
    expected_hash = meta.get("centroids_sha256")
    if expected_hash:
        actual_hash = hashlib.sha256(centroids.tobytes()).hexdigest()
        if actual_hash != expected_hash:
            return False, "Gallery centroids hash mismatch - file may be corrupted"

    return True, ""


def load_gallery(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Gallery file not found: {path}")

    data = np.load(path, allow_pickle=False)
    ok, reason = validate_gallery(data)
    if not ok:
        raise ValueError(f"Invalid gallery: {reason}")

    return (
        data["labels"],
        data["centroids"],
        data["sample_labels"],
        data["sample_embeddings"],
        json.loads(str(data["metadata"])),
    )
