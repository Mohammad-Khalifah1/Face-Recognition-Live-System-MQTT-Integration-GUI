from __future__ import annotations

import argparse
import json
from typing import Iterable

from rich.console import Console
from rich.table import Table

from face_id.api import DEFAULT_THRESHOLD, FaceRecognizer
from face_id.enroll import FaceEnroller
from face_id.live import LiveRecognizer
from face_id.app import AppController

console = Console()


def _build_gallery(args: argparse.Namespace) -> None:
    recognizer = FaceRecognizer(
        device=args.device,
        model=args.model,
        det_size=args.det_size,
        max_side=args.max_side,
    )
    result = recognizer.build_gallery(
        images_dir=args.images_dir,
        output=args.output,
        skip_quality=args.skip_quality_check,
        min_face_size=args.min_face_size,
        min_det_conf=args.min_det_conf,
    )

    table = Table(title="Gallery Built")
    table.add_column("Class")
    table.add_column("Accepted Images", justify="right")
    table.add_column("Quality Rejected", justify="right")
    table.add_column("Skipped", justify="right")
    for label in result["labels"]:
        accepted = result["per_person_counts"].get(label, 0)
        qr = len(result["quality_rejected"].get(label, []))
        sk = len(result["skipped"].get(label, []))
        table.add_row(label, str(accepted), str(qr), str(sk))
    console.print(table)

    if result["quality_rejected"]:
        console.print("\n[yellow]Quality-rejected images:[/yellow]")
        for person, reasons in result["quality_rejected"].items():
            for r in reasons:
                console.print(f"  {person}: {r}")

    if result["skipped"]:
        console.print("\n[yellow]Skipped (no face detected):[/yellow]")
        for person, reasons in result["skipped"].items():
            for r in reasons:
                console.print(f"  {person}: {r}")

    console.print(f"\n[green]Saved gallery:[/green] {result['output']}")


def _predict(args: argparse.Namespace) -> None:
    recognizer = FaceRecognizer(
        device=args.device,
        model=args.model,
        det_size=args.det_size,
        max_side=args.max_side,
    )
    results = recognizer.predict(
        input_path=args.input,
        gallery_path=args.gallery,
        threshold=args.threshold,
        recursive=args.recursive,
        all_faces=args.all_faces,
        face_index=args.face_index,
        matching=args.matching,
        knn_k=args.knn_k,
        verbose=args.verbose,
    )

    if args.json:
        console.print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
        return

    table = Table(title="Face Recognition Results")
    table.add_column("Image")
    table.add_column("Person")
    table.add_column("Similarity", justify="right")
    table.add_column("Accepted", justify="center")
    for r in results:
        table.add_row(
            r.image,
            r.person,
            f"{r.similarity_percent:.2f}%",
            "yes" if r.accepted else "no",
        )
    console.print(table)


def _calibrate(args: argparse.Namespace) -> None:
    recognizer = FaceRecognizer(
        device=args.device,
        model=args.model,
        det_size=args.det_size,
        max_side=args.max_side,
    )
    result = recognizer.calibrate(gallery_path=args.gallery)

    table = Table(title="Threshold Calibration Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Recommended Threshold", str(result["recommended_threshold"]))
    table.add_row("F1 Score", str(result["f1_score"]))
    table.add_row("Genuine Mean Similarity", str(result["genuine_mean"]))
    table.add_row("Impostor Mean Similarity", str(result["impostor_mean"]))
    table.add_row("Genuine Pairs", str(result["genuine_count"]))
    table.add_row("Impostor Pairs", str(result["impostor_count"]))
    console.print(table)

    console.print(
        f"\n[green]Tip:[/green] Use --threshold {result['recommended_threshold']} "
        f"with face-id predict for optimal accuracy."
    )


def _enroll(args: argparse.Namespace) -> None:
    enroller = FaceEnroller(
        name=args.name,
        camera_id=args.camera,
        images_dir=args.images_dir,
        target_count=args.count,
        min_face_size=args.min_face_size,
        min_det_conf=args.min_det_conf,
        device=args.device,
        model=args.model,
        det_size=args.det_size,
        max_side=args.max_side,
    )
    result = enroller.run()
    console.print(
        f"\n[green]Enrollment complete:[/green] "
        f"{result['captured']}/{result['target']} images"
    )
    console.print(f"[green]Saved to:[/green] {result['output_dir']}")


def _live(args: argparse.Namespace) -> None:
    recognizer = LiveRecognizer(
        gallery_path=args.gallery,
        camera_id=args.camera,
        device=args.device,
        model=args.model,
        det_size=args.det_size,
        max_side=args.max_side,
        threshold=args.threshold,
        matching=args.matching,
        knn_k=args.knn_k,
        stable_duration=args.stable_duration,
        display_duration=args.display_duration,
    )
    recognizer.run()


def _app(args: argparse.Namespace) -> None:
    mqtt_notifier = None
    if args.mqtt:
        from face_id.mqtt_client import MqttNotifier
        mqtt_notifier = MqttNotifier(
            broker=args.mqtt_broker,
            port=args.mqtt_port,
            topic_prefix=args.mqtt_topic,
        )
        mqtt_notifier.connect()

    controller = AppController(
        gallery_path=args.gallery,
        images_dir=args.images_dir,
        camera_id=args.camera,
        device=args.device,
        model=args.model,
        det_size=args.det_size,
        max_side=args.max_side,
        threshold=args.threshold,
        matching=args.matching,
        knn_k=args.knn_k,
        mqtt_notifier=mqtt_notifier,
    )
    controller.run()


def _add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--model", default="buffalo_s", help="InsightFace model pack name")
    parser.add_argument("--det-size", type=int, default=320, help="Face detector input size")
    parser.add_argument("--max-side", type=int, default=1280, help="Max image dimension before detection")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="face-id",
        description="Face recognition and classification without training from scratch.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build face gallery from images/<class> folders")
    build.add_argument("--images-dir", default="images")
    build.add_argument("--output", default="data/gallery.npz")
    build.add_argument("--skip-quality-check", action="store_true", help="Disable quality filtering")
    build.add_argument("--min-face-size", type=int, default=25)
    build.add_argument("--min-det-conf", type=float, default=0.5)
    _add_common_model_args(build)
    build.set_defaults(func=_build_gallery)

    pred = subparsers.add_parser("predict", help="Recognize faces in an image or folder")
    pred.add_argument("input")
    pred.add_argument("--gallery", default="data/gallery.npz")
    pred.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    pred.add_argument("--recursive", action="store_true")
    pred.add_argument("--all-faces", action="store_true", help="Detect and match all faces")
    pred.add_argument("--face-index", type=int, default=None, help="Select Nth face (0-indexed)")
    pred.add_argument("--matching", choices=["centroid", "knn"], default="centroid")
    pred.add_argument("--knn-k", type=int, default=3)
    pred.add_argument("--verbose", action="store_true", help="Show all match scores")
    pred.add_argument("--json", action="store_true")
    _add_common_model_args(pred)
    pred.set_defaults(func=_predict)

    cal = subparsers.add_parser("calibrate", help="Find optimal threshold via leave-one-out")
    cal.add_argument("--gallery", default="data/gallery.npz")
    _add_common_model_args(cal)
    cal.set_defaults(func=_calibrate)

    enr = subparsers.add_parser("enroll", help="Enroll new person from live camera")
    enr.add_argument("--name", required=True, help="Person name (English letters, digits, underscores)")
    enr.add_argument("--camera", type=int, default=0, help="Camera device ID")
    enr.add_argument("--images-dir", default="images")
    enr.add_argument("--count", type=int, default=10, help="Number of images to capture")
    enr.add_argument("--min-face-size", type=int, default=80)
    enr.add_argument("--min-det-conf", type=float, default=0.5)
    _add_common_model_args(enr)
    enr.set_defaults(func=_enroll)

    live = subparsers.add_parser("live", help="Real-time face recognition from camera")
    live.add_argument("--gallery", default="data/gallery.npz")
    live.add_argument("--camera", type=int, default=0, help="Camera device ID")
    live.add_argument("--threshold", type=float, default=0.45, help="Matching threshold (default: 0.45)")
    live.add_argument("--matching", choices=["centroid", "knn"], default="centroid")
    live.add_argument("--knn-k", type=int, default=3)
    live.add_argument("--stable-duration", type=float, default=2.0, help="Seconds of stability before predict")
    live.add_argument("--display-duration", type=float, default=3.0, help="Seconds to display result")
    _add_common_model_args(live)
    live.set_defaults(func=_live)

    app = subparsers.add_parser("app", help="Launch GUI application")
    app.add_argument("--gallery", default="data/gallery.npz")
    app.add_argument("--images-dir", default="images")
    app.add_argument("--camera", type=int, default=0)
    app.add_argument("--threshold", type=float, default=0.45)
    app.add_argument("--matching", choices=["centroid", "knn"], default="centroid")
    app.add_argument("--knn-k", type=int, default=3)
    app.add_argument("--mqtt", action="store_true", help="Enable MQTT notifications")
    app.add_argument("--mqtt-broker", default="localhost", metavar="IP",
                     help="MQTT broker address (default: localhost)")
    app.add_argument("--mqtt-port", type=int, default=1883,
                     help="MQTT broker port (default: 1883)")
    app.add_argument("--mqtt-topic", default="face-id", metavar="PREFIX",
                     help="MQTT topic prefix (default: face-id)")
    _add_common_model_args(app)
    app.set_defaults(func=_app)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
