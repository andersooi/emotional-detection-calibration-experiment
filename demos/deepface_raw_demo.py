"""
Raw DeepFace emotion demo.

Runs DeepFace on the full video frame using DeepFace's own face detector,
then displays:
- live video with the detected face box
- raw top emotion and confidence
- raw probabilities for all 7 DeepFace emotion classes

This is meant to isolate raw DeepFace behaviour from the calibration demos.

Usage:
    python demos/deepface_raw_demo.py --camera 1
    python demos/deepface_raw_demo.py --camera 1 --detector retinaface
    python demos/deepface_raw_demo.py --camera 1 --detector retinaface --analysis-width 384
"""

import time
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


EMOTION_LABELS = [
    "Anger",
    "Disgust",
    "Fear",
    "Happiness",
    "Sadness",
    "Surprise",
    "Neutral",
]

EMOTION_COLORS = {
    "Anger": "#E74C3C",
    "Disgust": "#8E44AD",
    "Fear": "#9B59B6",
    "Happiness": "#F1C40F",
    "Sadness": "#3498DB",
    "Surprise": "#E67E22",
    "Neutral": "#95A5A6",
}

COLORS = {
    "bg_dark": "#2C3E50",
    "bg_medium": "#34495E",
    "text_white": "#FFFFFF",
    "text_gray": "#BDC3C7",
    "accent_green": "#2ECC71",
    "accent_red": "#E74C3C",
    "accent_blue": "#3498DB",
    "accent_yellow": "#F1C40F",
}


class RawDeepFaceExtractor:
    """Runs DeepFace emotion analysis on full frames."""

    def __init__(self, detector_backend: str = "opencv", align: bool = True):
        self.detector_backend = detector_backend
        self.align = align
        self._deepface = None

    def load(self, status_callback=None):
        if status_callback:
            status_callback(f"Loading DeepFace ({self.detector_backend})...")

        from deepface import DeepFace

        self._deepface = DeepFace

        # Warm up only the emotion model. Detection will happen on live frames.
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        try:
            self._deepface.analyze(
                dummy,
                actions=["emotion"],
                detector_backend="skip",
                enforce_detection=False,
                silent=True,
            )
        except Exception:
            pass

        if status_callback:
            status_callback("DeepFace loaded!")

    def _is_full_frame_fallback(
        self, region: Dict, frame_shape: Tuple[int, int], face_confidence: float
    ) -> bool:
        """Detect DeepFace's no-face fallback region."""
        frame_h, frame_w = frame_shape
        x = int(region.get("x", 0) or 0)
        y = int(region.get("y", 0) or 0)
        w = int(region.get("w", 0) or 0)
        h = int(region.get("h", 0) or 0)

        return (
            x == 0
            and y == 0
            and abs(w - frame_w) <= 2
            and abs(h - frame_h) <= 2
            and face_confidence == 0.0
        )

    def _pick_best_face(
        self, analyses: List[Dict], frame_shape: Tuple[int, int]
    ) -> Optional[Dict]:
        """Select the largest valid detected face."""
        valid_faces = []

        for obj in analyses:
            region = obj.get("region") or {}
            face_confidence = float(obj.get("face_confidence", 0) or 0.0)
            w = int(region.get("w", 0) or 0)
            h = int(region.get("h", 0) or 0)

            if w <= 0 or h <= 0:
                continue

            if self._is_full_frame_fallback(region, frame_shape, face_confidence):
                continue

            valid_faces.append(obj)

        if not valid_faces:
            return None

        return max(
            valid_faces,
            key=lambda obj: (
                int((obj.get("region") or {}).get("w", 0) or 0)
                * int((obj.get("region") or {}).get("h", 0) or 0),
                float(obj.get("face_confidence", 0) or 0.0),
            ),
        )

    def analyze_frame(self, frame_bgr: np.ndarray) -> Dict:
        """Analyze a full BGR frame with DeepFace's own detector."""
        if self._deepface is None:
            self.load()

        analysis = self._deepface.analyze(
            frame_bgr,
            actions=["emotion"],
            detector_backend=self.detector_backend,
            align=self.align,
            enforce_detection=False,
            silent=True,
        )

        analyses = analysis if isinstance(analysis, list) else [analysis]
        best = self._pick_best_face(analyses, frame_bgr.shape[:2])

        if best is None:
            return {"detected": False}

        raw_probs = best.get("emotion", {})
        emotion_probs = {
            "Anger": raw_probs.get("angry", 0.0) / 100,
            "Disgust": raw_probs.get("disgust", 0.0) / 100,
            "Fear": raw_probs.get("fear", 0.0) / 100,
            "Happiness": raw_probs.get("happy", 0.0) / 100,
            "Sadness": raw_probs.get("sad", 0.0) / 100,
            "Surprise": raw_probs.get("surprise", 0.0) / 100,
            "Neutral": raw_probs.get("neutral", 0.0) / 100,
        }
        top_emotion = max(emotion_probs, key=emotion_probs.get)

        return {
            "detected": True,
            "emotion_probs": emotion_probs,
            "top_emotion": top_emotion,
            "confidence": emotion_probs[top_emotion],
            "region": best.get("region", {}),
            "face_confidence": float(best.get("face_confidence", 0) or 0.0),
        }


class DeepFaceRawDemoApp:
    """Standalone GUI for raw DeepFace live testing."""

    def __init__(
        self,
        camera_index: int = 0,
        detector_backend: str = "opencv",
        align: bool = True,
        analysis_width: int = 384,
        inference_interval_ms: int = 0,
    ):
        self.extractor = RawDeepFaceExtractor(
            detector_backend=detector_backend,
            align=align,
        )
        self.camera_index = camera_index
        self.detector_backend = detector_backend
        self.align = align
        self.analysis_width = max(0, int(analysis_width))
        self.inference_interval_ms = max(0, int(inference_interval_ms))

        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False

        self.state_lock = threading.Lock()
        self.inference_in_progress = False
        self.last_inference_launch = 0.0
        self.latest_bbox: Optional[Tuple[int, int, int, int]] = None

        self.inference_time = 0.0
        self.fps = 0.0
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.prediction_fps = 0.0
        self.prediction_count = 0
        self.prediction_fps_start = time.time()

        self.root = tk.Tk()
        self.root.title("DeepFace Raw Demo")
        self.root.geometry("1120x760")
        self.root.configure(bg=COLORS["bg_dark"])
        self._setup_ui()

    def _setup_ui(self):
        main = tk.Frame(self.root, bg=COLORS["bg_dark"])
        main.pack(fill="both", expand=True, padx=10, pady=10)

        top = tk.Frame(main, bg=COLORS["bg_dark"])
        top.pack(fill="x", pady=(0, 10))

        tk.Label(
            top,
            text="DeepFace Raw Demo",
            font=("Helvetica", 18, "bold"),
            bg=COLORS["bg_dark"],
            fg=COLORS["text_white"],
        ).pack(side="left")

        tk.Label(
            top,
            text=(
                f"Detector: {self.detector_backend} | Align: {'On' if self.align else 'Off'}"
                f" | Analysis width: {self.analysis_width if self.analysis_width > 0 else 'full'}"
                f" | Interval: {self.inference_interval_ms} ms"
            ),
            font=("Helvetica", 11),
            bg=COLORS["bg_dark"],
            fg=COLORS["text_gray"],
        ).pack(side="right")

        content = tk.Frame(main, bg=COLORS["bg_dark"])
        content.pack(fill="both", expand=True, pady=10)

        left = tk.Frame(content, bg=COLORS["bg_dark"])
        left.pack(side="left", fill="both", expand=True)

        self.video_label = tk.Label(left, bg="#1a1a2e")
        self.video_label.pack(pady=10)

        self.status_label = tk.Label(
            left,
            text="Initializing...",
            font=("Helvetica", 11),
            bg=COLORS["bg_dark"],
            fg=COLORS["text_gray"],
        )
        self.status_label.pack(pady=5)

        self.face_info_label = tk.Label(
            left,
            text="Face confidence: -- | Region: --",
            font=("Helvetica", 10),
            bg=COLORS["bg_dark"],
            fg=COLORS["text_gray"],
        )
        self.face_info_label.pack(pady=2)

        right = tk.Frame(content, bg=COLORS["bg_medium"], padx=15, pady=15)
        right.pack(side="right", fill="both", padx=(10, 0))

        tk.Label(
            right,
            text="Raw Prediction",
            font=("Helvetica", 14, "bold"),
            bg=COLORS["bg_medium"],
            fg=COLORS["text_white"],
        ).pack(pady=(0, 12))

        summary = tk.Frame(right, bg=COLORS["bg_dark"], padx=12, pady=12)
        summary.pack(fill="x", pady=(0, 12))

        tk.Label(
            summary,
            text="TOP EMOTION",
            font=("Helvetica", 10, "bold"),
            bg=COLORS["bg_dark"],
            fg=COLORS["accent_red"],
        ).pack()

        self.raw_emotion_label = tk.Label(
            summary,
            text="--",
            font=("Helvetica", 22, "bold"),
            bg=COLORS["bg_dark"],
            fg=COLORS["text_white"],
        )
        self.raw_emotion_label.pack(pady=8)

        self.raw_confidence_label = tk.Label(
            summary,
            text="Confidence: --",
            font=("Helvetica", 10),
            bg=COLORS["bg_dark"],
            fg=COLORS["text_gray"],
        )
        self.raw_confidence_label.pack()

        prob_frame = tk.Frame(right, bg=COLORS["bg_medium"])
        prob_frame.pack(fill="both", expand=True)

        tk.Label(
            prob_frame,
            text="RAW PROBABILITIES",
            font=("Helvetica", 10, "bold"),
            bg=COLORS["bg_medium"],
            fg=COLORS["text_white"],
        ).pack(pady=(0, 6))

        self.prob_bars = {}
        for label in EMOTION_LABELS:
            row = tk.Frame(prob_frame, bg=COLORS["bg_medium"])
            row.pack(fill="x", pady=2)

            tk.Label(
                row,
                text=label[:8],
                font=("Helvetica", 9),
                width=10,
                anchor="e",
                bg=COLORS["bg_medium"],
                fg=COLORS["text_gray"],
            ).pack(side="left")

            canvas = tk.Canvas(
                row,
                width=180,
                height=16,
                bg="#1e293b",
                highlightthickness=0,
            )
            canvas.pack(side="left", padx=(8, 2))

            value_label = tk.Label(
                row,
                text="--",
                font=("Helvetica", 8),
                width=5,
                bg=COLORS["bg_medium"],
                fg=COLORS["text_gray"],
            )
            value_label.pack(side="left")

            self.prob_bars[label] = {
                "canvas": canvas,
                "label": value_label,
                "color": EMOTION_COLORS[label],
            }

        bottom = tk.Frame(main, bg=COLORS["bg_dark"])
        bottom.pack(fill="x", pady=(10, 0))

        self.metrics_label = tk.Label(
            bottom,
            text="Latency: -- ms | Camera FPS: -- | Predictions/s: --",
            font=("Helvetica", 10),
            bg=COLORS["bg_dark"],
            fg=COLORS["text_gray"],
        )
        self.metrics_label.pack(side="right")

    def _resize_for_analysis(
        self, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """Optionally downscale frame before DeepFace analysis."""
        if self.analysis_width <= 0 or frame_bgr.shape[1] <= self.analysis_width:
            return frame_bgr, 1.0, 1.0

        scale = self.analysis_width / frame_bgr.shape[1]
        resized_h = max(1, int(round(frame_bgr.shape[0] * scale)))
        resized = cv2.resize(frame_bgr, (self.analysis_width, resized_h))
        scale_x = frame_bgr.shape[1] / resized.shape[1]
        scale_y = frame_bgr.shape[0] / resized.shape[0]
        return resized, scale_x, scale_y

    def _scale_region(
        self,
        region: Dict,
        scale_x: float,
        scale_y: float,
        frame_shape: Tuple[int, int, int],
    ) -> Dict:
        """Scale a detected region back to the original frame size."""
        frame_h, frame_w = frame_shape[:2]
        x = int(round((region.get("x", 0) or 0) * scale_x))
        y = int(round((region.get("y", 0) or 0) * scale_y))
        w = int(round((region.get("w", 0) or 0) * scale_x))
        h = int(round((region.get("h", 0) or 0) * scale_y))

        x = max(0, min(frame_w - 1, x))
        y = max(0, min(frame_h - 1, y))
        w = max(0, min(frame_w - x, w))
        h = max(0, min(frame_h - y, h))

        scaled = dict(region)
        scaled["x"] = x
        scaled["y"] = y
        scaled["w"] = w
        scaled["h"] = h
        return scaled

    def _clear_display(self):
        """Clear prediction displays when no valid face is available."""
        self.raw_emotion_label.config(text="NO FACE")
        self.raw_confidence_label.config(text="Confidence: --")
        self.face_info_label.config(text="Face confidence: -- | Region: --")
        for bars in self.prob_bars.values():
            bars["canvas"].delete("all")
            bars["label"].config(text="--")

    def _handle_inference_result(self, result: Dict):
        """Apply a completed inference result to the UI."""
        if result.get("detected"):
            self.update_display(result)
            self.status_label.config(text="Face detected. Showing raw DeepFace output.")
        else:
            self._clear_display()
            self.status_label.config(text="No valid face detected by DeepFace.")

    def _run_inference(self, frame_bgr: np.ndarray):
        """Run DeepFace inference in the background on the latest frame."""
        try:
            analysis_frame, scale_x, scale_y = self._resize_for_analysis(frame_bgr)
            start = time.time()
            result = self.extractor.analyze_frame(analysis_frame)
            latency = time.time() - start

            if result.get("detected"):
                result["region"] = self._scale_region(
                    result.get("region", {}),
                    scale_x,
                    scale_y,
                    frame_bgr.shape,
                )
                region = result["region"]
                bbox = (
                    int(region.get("x", 0) or 0),
                    int(region.get("y", 0) or 0),
                    int((region.get("x", 0) or 0) + (region.get("w", 0) or 0)),
                    int((region.get("y", 0) or 0) + (region.get("h", 0) or 0)),
                )
            else:
                bbox = None

            with self.state_lock:
                self.inference_time = latency
                self.latest_bbox = bbox
                self.inference_in_progress = False

                self.prediction_count += 1
                pred_elapsed = time.time() - self.prediction_fps_start
                if pred_elapsed >= 1.0:
                    self.prediction_fps = self.prediction_count / pred_elapsed
                    self.prediction_count = 0
                    self.prediction_fps_start = time.time()

            self.root.after(0, lambda r=result: self._handle_inference_result(r))

        except Exception as exc:
            err = str(exc)
            print(f"Inference error: {err}")
            import traceback

            traceback.print_exc()
            with self.state_lock:
                self.inference_in_progress = False
                self.latest_bbox = None
            self.root.after(0, self._clear_display)
            self.root.after(
                0,
                lambda e=err: self.status_label.config(text=f"Error: {e}"),
            )

    def update_video(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None):
        if frame is None:
            return

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (520, 390))
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def update_display(self, result: Dict):
        self.raw_emotion_label.config(text=result["top_emotion"])
        self.raw_confidence_label.config(text=f"Confidence: {result['confidence']:.0%}")

        region = result.get("region", {})
        x = int(region.get("x", 0) or 0)
        y = int(region.get("y", 0) or 0)
        w = int(region.get("w", 0) or 0)
        h = int(region.get("h", 0) or 0)
        face_confidence = float(result.get("face_confidence", 0) or 0.0)
        self.face_info_label.config(
            text=f"Face confidence: {face_confidence:.2f} | Region: x={x}, y={y}, w={w}, h={h}"
        )

        probs = result.get("emotion_probs", {})
        for label, bars in self.prob_bars.items():
            p = float(probs.get(label, 0.0))
            bars["canvas"].delete("all")
            width = max(0, min(180, int(180 * p)))
            if width > 0:
                bars["canvas"].create_rectangle(
                    0, 0, width, 16, fill=bars["color"], outline=""
                )
            bars["label"].config(text=f"{p:.0%}")

    def main_loop(self):
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(0.1)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                with self.state_lock:
                    bbox = self.latest_bbox
                    can_launch = not self.inference_in_progress
                    last_launch = self.last_inference_launch
                    inference_time = self.inference_time
                    prediction_fps = self.prediction_fps

                now = time.time()
                due = (
                    self.inference_interval_ms <= 0
                    or (now - last_launch) * 1000 >= self.inference_interval_ms
                )
                if can_launch and due:
                    with self.state_lock:
                        self.inference_in_progress = True
                        self.last_inference_launch = now
                    threading.Thread(
                        target=self._run_inference,
                        args=(frame.copy(),),
                        daemon=True,
                    ).start()

                self.root.after(0, lambda f=frame.copy(), b=bbox: self.update_video(f, b))

                self.frame_count += 1
                elapsed = time.time() - self.fps_start_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.fps_start_time = time.time()
                    self.root.after(
                        0,
                        lambda it=inference_time, pf=prediction_fps: self.metrics_label.config(
                            text=(
                                f"Latency: {it * 1000:.0f} ms"
                                f" | Camera FPS: {self.fps:.1f}"
                                f" | Predictions/s: {pf:.1f}"
                            )
                        ),
                    )

                time.sleep(0.01)
            except Exception as exc:
                err = str(exc)
                print(f"Loop error: {err}")
                import traceback

                traceback.print_exc()
                self.root.after(
                    0,
                    lambda e=err: self.status_label.config(text=f"Error: {e}"),
                )
                time.sleep(0.2)

    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        def init():
            self.root.after(
                0,
                lambda: self.status_label.config(text="Loading DeepFace..."),
            )
            self.extractor.load(
                status_callback=lambda msg: self.root.after(
                    0, lambda m=msg: self.status_label.config(text=m)
                )
            )

            self.root.after(
                0,
                lambda: self.status_label.config(
                    text=f"Starting camera {self.camera_index}..."
                ),
            )
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            time.sleep(0.5)

            self.running = True
            self.root.after(
                0,
                lambda: self.status_label.config(
                    text="Ready. Testing raw DeepFace on full frames."
                ),
            )
            threading.Thread(target=self.main_loop, daemon=True).start()

        threading.Thread(target=init, daemon=True).start()
        self.root.mainloop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepFace Raw Demo")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (0=default, 1=MacBook webcam)",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="opencv",
        help="DeepFace detector backend (e.g. opencv, retinaface, mediapipe, mtcnn)",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Disable DeepFace face alignment",
    )
    parser.add_argument(
        "--analysis-width",
        type=int,
        default=384,
        help="Resize frame width before analysis (0=full resolution)",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=0,
        help="Minimum delay between DeepFace inferences in milliseconds",
    )
    args = parser.parse_args()

    print(f"Using camera index: {args.camera}")
    print(f"Using detector backend: {args.detector}")
    print(f"Alignment: {'off' if args.no_align else 'on'}")
    print(
        f"Analysis width: {args.analysis_width if args.analysis_width > 0 else 'full'}"
    )
    print(f"Inference interval: {args.interval_ms} ms")

    app = DeepFaceRawDemoApp(
        camera_index=args.camera,
        detector_backend=args.detector,
        align=not args.no_align,
        analysis_width=args.analysis_width,
        inference_interval_ms=args.interval_ms,
    )
    app.run()
