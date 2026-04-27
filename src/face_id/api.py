from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from face_id.gallery import GALLERY_VERSION, build_metadata, load_gallery, save_gallery
from face_id.matching import match_all_scores, match_embedding, match_embedding_knn
from face_id.quality import check_face_quality

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_THRESHOLD = 0.38


@dataclass(frozen=True)
class PredictResult:
    image: str
    person: str
    similarity: float
    similarity_percent: float
    accepted: bool
    all_scores: list[dict] | None = None

    def to_dict(self) -> dict:
        d = {
            "image": self.image,
            "person": self.person,
            "similarity": round(self.similarity, 4),
            "similarity_percent": round(self.similarity_percent, 2),
            "accepted": self.accepted,
        }
        if self.all_scores is not None:
            d["all_scores"] = self.all_scores
        return d


class FaceRecognizer:
    def __init__(
        self,
        device: str = "gpu",
        model: str = "buffalo_s",
        det_size: int = 320,
        max_side: int = 1280,
    ) -> None:
        self.device = device
        self.model_name = model
        self.det_size = det_size
        self.max_side = max_side
        self._app: FaceAnalysis | None = None

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

    @staticmethod
    def image_paths(path: Path, recursive: bool = False) -> list[Path]:
        if path.is_file():
            return [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []
        pattern = "**/*" if recursive else "*"
        return sorted(
            item
            for item in path.glob(pattern)
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )

    def load_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest <= self.max_side:
            return image
        scale = self.max_side / float(longest)
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def _choose_largest_face(self, faces: list) -> object | None:
        if not faces:
            return None
        return max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))

    def _embedding_for_image(self, path: Path) -> np.ndarray | None:
        image = self.load_image(path)
        face = self._choose_largest_face(self.app.get(image))
        if face is None:
            return None
        emb = getattr(face, "normed_embedding", None)
        if emb is not None:
            return np.asarray(emb, dtype=np.float32)
        raw = np.asarray(face.embedding, dtype=np.float32)
        norm = np.linalg.norm(raw)
        return raw / norm if norm != 0 else raw

    def _all_faces_embeddings(self, path: Path) -> list[tuple[object, np.ndarray]]:
        image = self.load_image(path)
        faces = self.app.get(image)
        results = []
        for face in faces:
            emb = getattr(face, "normed_embedding", None)
            if emb is not None:
                results.append((face, np.asarray(emb, dtype=np.float32)))
            else:
                raw = np.asarray(face.embedding, dtype=np.float32)
                norm = np.linalg.norm(raw)
                if norm != 0:
                    results.append((face, raw / norm))
        return results

    @staticmethod
    def _class_dirs(images_dir: Path) -> list[Path]:
        return sorted(item for item in images_dir.iterdir() if item.is_dir())

    def build_gallery(
        self,
        images_dir: str | Path,
        output: str | Path = "data/gallery.npz",
        skip_quality: bool = False,
        min_face_size: int = 80,
        min_det_conf: float = 0.5,
    ) -> dict:
        from tqdm import tqdm

        images_dir = Path(images_dir)
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory does not exist: {images_dir}")

        labels: list[str] = []
        centroids: list[np.ndarray] = []
        sample_embeddings: list[np.ndarray] = []
        sample_labels: list[str] = []
        skipped: dict[str, list[str]] = {}
        quality_rejected: dict[str, list[str]] = {}

        dirs = self._class_dirs(images_dir)
        if not dirs:
            raise ValueError(f"No class folders found inside: {images_dir}")

        for person_dir in dirs:
            paths = self.image_paths(person_dir)
            if not paths:
                skipped[person_dir.name] = ["No images found"]
                continue

            person_embeddings: list[np.ndarray] = []
            for img_path in tqdm(paths, desc=f"Building {person_dir.name}", unit="img"):
                image = self.load_image(img_path)
                faces = self.app.get(image)
                face = self._choose_largest_face(faces)

                if face is None:
                    skipped.setdefault(person_dir.name, []).append(f"no face: {img_path.name}")
                    continue

                if not skip_quality:
                    ok, reason = check_face_quality(
                        face, image, min_face_size=min_face_size, min_det_conf=min_det_conf
                    )
                    if not ok:
                        quality_rejected.setdefault(person_dir.name, []).append(
                            f"{img_path.name}: {reason}"
                        )
                        continue

                emb = getattr(face, "normed_embedding", None)
                if emb is not None:
                    emb = np.asarray(emb, dtype=np.float32)
                else:
                    raw = np.asarray(face.embedding, dtype=np.float32)
                    norm = np.linalg.norm(raw)
                    emb = raw / norm if norm != 0 else raw

                person_embeddings.append(emb)
                sample_embeddings.append(emb)
                sample_labels.append(person_dir.name)

            if person_embeddings:
                labels.append(person_dir.name)
                avg = np.mean(person_embeddings, axis=0).astype(np.float32)
                norm = np.linalg.norm(avg)
                if norm != 0:
                    avg = avg / norm
                centroids.append(avg)

        if not centroids:
            raise ValueError("No faces were detected. Add clear face images and run build again.")

        output = Path(output)
        meta = build_metadata(
            images_dir=str(images_dir),
            model=self.model_name,
            det_size=self.det_size,
            max_side=self.max_side,
            labels=labels,
            skipped=skipped,
        )
        save_gallery(output, labels, centroids, sample_labels, sample_embeddings, meta)

        return {
            "output": str(output),
            "labels": labels,
            "skipped": skipped,
            "quality_rejected": quality_rejected,
            "per_person_counts": {
                label: sum(1 for s in sample_labels if s == label) for label in labels
            },
        }

    def predict(
        self,
        input_path: str | Path,
        gallery_path: str | Path = "data/gallery.npz",
        threshold: float = DEFAULT_THRESHOLD,
        recursive: bool = False,
        all_faces: bool = False,
        face_index: int | None = None,
        matching: str = "centroid",
        knn_k: int = 3,
        verbose: bool = False,
    ) -> list[PredictResult]:
        from tqdm import tqdm

        gallery_path = Path(gallery_path)
        labels, centroids, sample_labels, sample_embeddings, meta = load_gallery(gallery_path)

        input_path = Path(input_path)
        paths = self.image_paths(input_path, recursive=recursive)
        if not paths:
            raise FileNotFoundError(f"No input images found at: {input_path}")

        results: list[PredictResult] = []
        for img_path in tqdm(paths, desc="Predicting", unit="img"):
            if all_faces:
                face_embs = self._all_faces_embeddings(img_path)
                if not face_embs:
                    results.append(PredictResult(
                        image=str(img_path), person="no_face_detected",
                        similarity=0.0, similarity_percent=0.0, accepted=False,
                    ))
                    continue
                for face_obj, emb in face_embs:
                    result = self._match_one(
                        emb, labels, centroids, sample_labels, sample_embeddings,
                        threshold, matching, knn_k, verbose,
                    )
                    results.append(PredictResult(
                        image=str(img_path), **result,
                    ))
            else:
                if face_index is not None:
                    image = self.load_image(img_path)
                    faces = self.app.get(image)
                    if face_index >= len(faces):
                        results.append(PredictResult(
                            image=str(img_path), person="face_index_out_of_range",
                            similarity=0.0, similarity_percent=0.0, accepted=False,
                        ))
                        continue
                    face = faces[face_index]
                    emb = getattr(face, "normed_embedding", None)
                    if emb is not None:
                        emb = np.asarray(emb, dtype=np.float32)
                    else:
                        raw = np.asarray(face.embedding, dtype=np.float32)
                        norm = np.linalg.norm(raw)
                        emb = raw / norm if norm != 0 else raw
                else:
                    emb = self._embedding_for_image(img_path)

                if emb is None:
                    results.append(PredictResult(
                        image=str(img_path), person="no_face_detected",
                        similarity=0.0, similarity_percent=0.0, accepted=False,
                    ))
                    continue

                result = self._match_one(
                    emb, labels, centroids, sample_labels, sample_embeddings,
                    threshold, matching, knn_k, verbose,
                )
                results.append(PredictResult(image=str(img_path), **result))

        return results

    @staticmethod
    def _match_one(
        emb: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
        sample_labels: np.ndarray,
        sample_embeddings: np.ndarray,
        threshold: float,
        matching: str,
        knn_k: int,
        verbose: bool,
    ) -> dict:
        if matching == "knn":
            m = match_embedding_knn(emb, sample_labels, sample_embeddings, k=knn_k, threshold=threshold)
        else:
            m = match_embedding(emb, labels, centroids, threshold)

        all_scores = None
        if verbose and matching == "centroid":
            all_scores = match_all_scores(emb, labels, centroids)

        return {
            "person": m["label"],
            "similarity": m["similarity"],
            "similarity_percent": m["percent"],
            "accepted": m["accepted"],
            "all_scores": all_scores,
        }

    def calibrate(self, gallery_path: str | Path = "data/gallery.npz") -> dict:
        gallery_path = Path(gallery_path)
        labels, centroids, sample_labels, sample_embeddings, meta = load_gallery(gallery_path)

        unique_labels = list(set(str(l) for l in sample_labels))
        if len(unique_labels) < 2:
            raise ValueError("Calibration requires at least 2 people in the gallery")

        genuine_scores: list[float] = []
        impostor_scores: list[float] = []

        for i in range(len(sample_labels)):
            emb = sample_embeddings[i]
            label = str(sample_labels[i])
            scores = centroids @ emb
            for j, sc in enumerate(scores):
                matched_label = str(labels[j])
                if matched_label == label:
                    genuine_scores.append(float(sc))
                else:
                    impostor_scores.append(float(sc))

        best_threshold = DEFAULT_THRESHOLD
        best_f1 = 0.0
        for t in np.arange(0.20, 0.65, 0.01):
            tp = sum(1 for s in genuine_scores if s >= t)
            fn = sum(1 for s in genuine_scores if s < t)
            fp = sum(1 for s in impostor_scores if s >= t)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(t)

        return {
            "recommended_threshold": round(best_threshold, 2),
            "f1_score": round(best_f1, 4),
            "genuine_mean": round(float(np.mean(genuine_scores)), 4),
            "impostor_mean": round(float(np.mean(impostor_scores)), 4),
            "genuine_count": len(genuine_scores),
            "impostor_count": len(impostor_scores),
        }
