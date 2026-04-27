from __future__ import annotations

import base64
import shutil
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path

import cv2

from face_id.api import FaceRecognizer
from face_id.command_executor import CommandExecutor
from face_id.enroll import FaceEnroller, validate_name
from face_id.live import LiveRecognizer
from face_id.mqtt_client import MqttNotifier


class AppController:
    def __init__(
        self,
        gallery_path: str = "data/gallery.npz",
        images_dir: str = "images",
        camera_id: int = 0,
        device: str = "gpu",
        model: str = "buffalo_s",
        det_size: int = 320,
        max_side: int = 1280,
        threshold: float = 0.45,
        matching: str = "centroid",
        knn_k: int = 3,
        mqtt_notifier: MqttNotifier | None = None,
    ) -> None:
        self.gallery_path = gallery_path
        self.images_dir = Path(images_dir)
        self.camera_id = camera_id
        self.device = device
        self.model = model
        self.det_size = det_size
        self.max_side = max_side
        self.threshold = threshold
        self.matching = matching
        self.knn_k = knn_k
        self.command_executor = CommandExecutor()
        self._mqtt = mqtt_notifier

        self._cap: cv2.VideoCapture | None = None
        self._mode: str = "none"
        self._model_thread: threading.Thread | None = None
        self._detection_thread: threading.Thread | None = None
        self._stop_detection: threading.Event = threading.Event()
        self._frame_pending: bool = False
        self._live: LiveRecognizer | None = None
        self._enroller: FaceEnroller | None = None
        self._building = False

        self.root = tk.Tk()
        self.root.title("Face Recognition System")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_exit)

        self._build_ui()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        # Left: camera feed — fixed 480×360 container
        cam_frame = tk.Frame(outer, bg="black", width=480, height=360)
        cam_frame.pack_propagate(False)
        cam_frame.pack(side=tk.LEFT, padx=(0, 10))
        self._camera_label = tk.Label(
            cam_frame, bg="black", text="Loading...", fg="white",
            font=("Helvetica", 14),
        )
        self._camera_label.pack(expand=True, fill=tk.BOTH)

        # Right: controls
        panel = ttk.Frame(outer, padding=(0, 5))
        panel.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(panel, text="Face Recognition System", font=("Helvetica", 14, "bold")).pack(
            pady=(0, 12)
        )

        status_frame = ttk.LabelFrame(panel, text="Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        self._status_var = tk.StringVar(value="Starting...")
        ttk.Label(status_frame, textvariable=self._status_var, font=("Helvetica", 11),
                  wraplength=220).pack()

        self._gallery_var = tk.StringVar(value="Gallery: loading...")
        ttk.Label(status_frame, textvariable=self._gallery_var, font=("Helvetica", 10)).pack()

        btn_frame = ttk.Frame(panel)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(btn_frame, text="Add Person", command=self._show_enroll_dialog).pack(
            fill=tk.X, pady=3
        )
        ttk.Button(btn_frame, text="Manage People", command=self._show_manage_dialog).pack(
            fill=tk.X, pady=3
        )
        ttk.Button(btn_frame, text="Rebuild Gallery", command=self._rebuild_gallery).pack(
            fill=tk.X, pady=3
        )
        ttk.Button(btn_frame, text="Exit", command=self._on_exit).pack(fill=tk.X, pady=(10, 3))

        self._update_gallery_info()

    def _show_frame(self, frame: "cv2.Mat") -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(640 / w, 480 / h)
        if scale != 1.0:
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
        h2, w2 = rgb.shape[:2]
        _, buf = cv2.imencode(".png", rgb)
        photo = tk.PhotoImage(data=base64.b64encode(buf).decode("ascii"))
        self._camera_label.configure(image=photo, text="", width=w2, height=h2)
        self._camera_label.image = photo

    def _update_gallery_info(self) -> None:
        if self.images_dir.exists():
            people = [
                d.name
                for d in self.images_dir.iterdir()
                if d.is_dir() and any(f.suffix.lower() in {".jpg", ".jpeg", ".png"} for f in d.iterdir())
            ]
            self._gallery_var.set(f"Gallery: {len(people)} person(s)")
        else:
            self._gallery_var.set("Gallery: 0 person(s)")

    # --- Live Recognition ---

    def _start_live(self) -> None:
        self._stop_camera()
        self._status_var.set("Loading model...")
        try:
            live = LiveRecognizer(
                gallery_path=self.gallery_path,
                camera_id=self.camera_id,
                device=self.device,
                model=self.model,
                det_size=self.det_size,
                max_side=self.max_side,
                threshold=self.threshold,
                matching=self.matching,
                knn_k=self.knn_k,
                on_recognized=self._on_recognized,
            )
        except Exception as e:
            self._status_var.set(f"Error loading gallery: {e}")
            return
        self._live = live

        def _load_model() -> None:
            _ = live.app
            self.root.after(0, self._open_live_camera)

        self._model_thread = threading.Thread(target=_load_model, daemon=True)
        self._model_thread.start()

    def _open_live_camera(self) -> None:
        if self._live is None:
            return
        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            self._status_var.set("Could not open camera")
            self._cap = None
            return
        self._mode = "live"
        self._status_var.set("Live recognition active")
        self._stop_detection.clear()
        self._frame_pending = False
        self._detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._detection_thread.start()

    def _on_recognized(self, name: str, similarity: float, accepted: bool) -> None:
        self.root.after(0, self._update_recognition_status, name, similarity, accepted)
        if accepted:
            self.command_executor.execute(name, similarity)
            if self._mqtt is not None:
                self._mqtt.publish_approved(name, similarity)
        else:
            if self._mqtt is not None:
                self._mqtt.publish_rejected()

    def _update_recognition_status(self, name: str, similarity: float, accepted: bool) -> None:
        if accepted:
            self._status_var.set(f"Recognized: {name} ({similarity:.1f}%)")
        else:
            self._status_var.set("Unknown person")

    # --- Enrollment ---

    def _show_enroll_dialog(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Add New Person")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Add New Person", font=("Helvetica", 14, "bold")).pack(pady=(0, 10))

        name_frame = ttk.Frame(frame)
        name_frame.pack(fill=tk.X, pady=5)
        ttk.Label(name_frame, text="Name:").pack(side=tk.LEFT)
        name_entry = ttk.Entry(name_frame, width=25)
        name_entry.pack(side=tk.LEFT, padx=(5, 0))

        count_frame = ttk.Frame(frame)
        count_frame.pack(fill=tk.X, pady=5)
        ttk.Label(count_frame, text="Images:").pack(side=tk.LEFT)
        count_var = tk.IntVar(value=10)
        ttk.Spinbox(count_frame, from_=3, to=30, textvariable=count_var, width=5).pack(
            side=tk.LEFT, padx=(5, 0)
        )

        status_label = ttk.Label(frame, text="", foreground="orange")
        status_label.pack(pady=5)

        def do_start() -> None:
            name = name_entry.get().strip()
            if not name:
                status_label.config(text="Enter a name", foreground="red")
                return
            try:
                validate_name(name)
            except ValueError as e:
                status_label.config(text=str(e), foreground="red")
                return

            dialog.destroy()
            self._start_enroll(name, count_var.get())

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(btn_frame, text="Start Capture", command=do_start).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5)
        )
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0)
        )

        name_entry.focus_set()
        dialog.bind("<Return>", lambda e: do_start())

    def _start_enroll(self, name: str, count: int) -> None:
        self._stop_camera()
        self._status_var.set(f"Loading model for '{name}'...")

        enroller = FaceEnroller(
            name=name,
            camera_id=self.camera_id,
            images_dir=self.images_dir,
            target_count=count,
            device=self.device,
            model=self.model,
            det_size=self.det_size,
            max_side=self.max_side,
        )
        self._enroller = enroller

        def _load_model() -> None:
            enroller.begin_session()
            _ = enroller.app
            self.root.after(0, self._open_enroll_camera)

        self._model_thread = threading.Thread(target=_load_model, daemon=True)
        self._model_thread.start()

    def _open_enroll_camera(self) -> None:
        if self._enroller is None:
            return
        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            self._status_var.set("Could not open camera")
            self._cap = None
            return
        self._mode = "enroll"
        self._status_var.set(f"Enrolling '{self._enroller.name}'...")
        self._stop_detection.clear()
        self._frame_pending = False
        self._detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._detection_thread.start()

    def _detection_loop(self) -> None:
        while not self._stop_detection.is_set():
            cap = self._cap
            if cap is None or not cap.isOpened():
                break
            ret, frame = cap.read()
            if not ret:
                continue

            mode = self._mode
            if mode == "live" and self._live is not None:
                display, stop = self._live.process_frame(frame, 0)
                self._push_frame(display)
                if stop:
                    self.root.after(0, self._on_live_stopped)
                    break

            elif mode == "enroll" and self._enroller is not None:
                display, done, result = self._enroller.process_frame(frame, 0)
                self._push_frame(display)
                if done:
                    r = result
                    self.root.after(0, self._on_enroll_done, r)
                    break

    def _push_frame(self, frame: "cv2.Mat") -> None:
        if self._frame_pending:
            return
        self._frame_pending = True
        h, w = frame.shape[:2]
        scale = min(480 / w, 360 / h)
        if scale != 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        self.root.after(0, self._update_label, buf.tobytes())

    def _update_label(self, png_bytes: bytes) -> None:
        photo = tk.PhotoImage(data=base64.b64encode(png_bytes).decode("ascii"))
        self._camera_label.configure(image=photo, text="")
        self._camera_label.image = photo
        self._frame_pending = False

    def _on_live_stopped(self) -> None:
        self._stop_camera()
        self._status_var.set("Camera stopped.")

    def _on_enroll_done(self, result: dict | None) -> None:
        self._stop_camera()
        if result is not None:
            self._post_enroll(result)

    def _post_enroll(self, result: dict) -> None:
        captured = result["captured"]
        name = result["name"]
        self._status_var.set(f"Enrolled '{name}': {captured} images. Building gallery...")

        if captured > 0:
            self._do_build_gallery()
        self._update_gallery_info()
        self._start_live()

    # --- Manage People ---

    def _show_manage_dialog(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Manage People")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Manage People", font=("Helvetica", 14, "bold")).pack(pady=(0, 10))

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        listbox = tk.Listbox(
            list_frame, width=35, height=10, yscrollcommand=scrollbar.set, font=("Helvetica", 11)
        )
        scrollbar.config(command=listbox.yview)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        people = self._get_people()
        for person, count in people:
            listbox.insert(tk.END, f"{person}  ({count} images)")

        status_label = ttk.Label(frame, text="", foreground="orange")
        status_label.pack(pady=5)

        def do_delete() -> None:
            sel = listbox.curselection()
            if not sel:
                status_label.config(text="Select a person first", foreground="red")
                return
            idx = sel[0]
            name = people[idx][0]
            if messagebox.askyesno("Confirm Delete", f"Delete '{name}' and all their images?"):
                person_dir = self.images_dir / name
                if person_dir.exists():
                    shutil.rmtree(person_dir)
                listbox.delete(idx)
                people[:] = people[:idx] + people[idx + 1 :]
                status_label.config(text=f"Deleted '{name}'. Rebuild gallery.", foreground="green")

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(btn_frame, text="Delete Selected", command=do_delete).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5)
        )
        ttk.Button(btn_frame, text="Close", command=dialog.destroy).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0)
        )

    def _get_people(self) -> list[tuple[str, int]]:
        result = []
        if not self.images_dir.exists():
            return result
        for d in sorted(self.images_dir.iterdir()):
            if d.is_dir():
                count = sum(1 for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
                if count > 0:
                    result.append((d.name, count))
        return result

    # --- Gallery Build ---

    def _rebuild_gallery(self) -> None:
        self._stop_camera()
        self._status_var.set("Building gallery...")
        self._do_build_gallery()
        self._update_gallery_info()
        self._start_live()

    def _do_build_gallery(self) -> None:
        recognizer = FaceRecognizer(
            device=self.device,
            model=self.model,
            det_size=self.det_size,
            max_side=self.max_side,
        )
        try:
            recognizer.build_gallery(
                images_dir=self.images_dir,
                output=self.gallery_path,
            )
            self._status_var.set("Gallery built successfully")
        except Exception as e:
            self._status_var.set(f"Build error: {e}")

    # --- Camera Control ---

    def _stop_camera(self) -> None:
        self._mode = "none"
        self._stop_detection.set()
        if self._live is not None:
            self._live.stop()
            self._live = None
        if self._enroller is not None:
            self._enroller.stop()
            self._enroller = None
        if self._detection_thread is not None and self._detection_thread.is_alive():
            self._detection_thread.join(timeout=3)
            self._detection_thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._model_thread is not None and self._model_thread.is_alive():
            self._model_thread.join(timeout=3)
            self._model_thread = None
        self._frame_pending = False
        if hasattr(self, "_camera_label"):
            self._camera_label.configure(image="", text="Loading...")
            self._camera_label.image = None

    # --- Lifecycle ---

    def _on_exit(self) -> None:
        self._stop_camera()
        if self._mqtt is not None:
            self._mqtt.disconnect()
        self.root.destroy()

    def run(self) -> None:
        self._start_live()
        self.root.mainloop()
