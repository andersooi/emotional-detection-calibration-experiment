"""
DeepFace Calibration Test Demo

Shows raw model output vs calibrated output side-by-side using DeepFace.
No valence-arousal — purely embedding-based calibration.

Usage:
    python demos/deepface_demo.py --camera 1
"""

import time
import threading
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
from typing import Optional, List, Dict

from core import (
    DeepFaceExtractor,
    GenericBaseline,
    GenericCalibrationManager,
    GenericCalibratedDetector,
    average_embeddings,
)


# ============================================================================
# Configuration
# ============================================================================

FRAMES_TO_AVERAGE = 25

CALIBRATION_STATES = [
    {'name': 'neutral', 'label': 'Neutral', 'duration': 5,
     'instruction': 'Look at the camera with a relaxed, natural expression.'},
    {'name': 'happy', 'label': 'Happy', 'duration': 5,
     'instruction': 'Think of a happy memory. Let yourself smile naturally.'},
    {'name': 'calm', 'label': 'Calm', 'duration': 5,
     'instruction': 'Take a deep breath. Feel relaxed and peaceful.'},
]

COLORS = {
    'bg_dark': '#2C3E50',
    'bg_medium': '#34495E',
    'text_white': '#FFFFFF',
    'text_gray': '#BDC3C7',
    'accent_green': '#2ECC71',
    'accent_red': '#E74C3C',
    'accent_blue': '#3498DB',
    'accent_yellow': '#F1C40F',
}


# ============================================================================
# Face Detector
# ============================================================================

class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return None, None
        areas = [w * h for (x, y, w, h) in faces]
        idx = np.argmax(areas)
        x, y, w, h = faces[idx]
        margin = int(0.1 * w)
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)
        face_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        return face_rgb, (x1, y1, x2, y2)


# ============================================================================
# Main Application
# ============================================================================

class DeepFaceDemoApp:
    """GUI for testing DeepFace calibration."""

    def __init__(self, camera_index: int = 0):
        self.extractor = DeepFaceExtractor(model_name='VGG-Face')
        self.face_detector = FaceDetector()
        self.cal_manager = GenericCalibrationManager(modality='deepface')
        self.detector = GenericCalibratedDetector()

        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.current_user: Optional[str] = None

        # Calibration state
        self.cal_in_progress = False
        self.cal_state_idx = 0
        self.cal_start_time = 0.0
        self.captured_frames: List[Dict] = []
        self._temp_baseline: Optional[GenericBaseline] = None

        # Metrics
        self.inference_time = 0.0
        self.fps = 0.0
        self.frame_count = 0
        self.fps_start_time = time.time()

        # GUI
        self.root = tk.Tk()
        self.root.title("DeepFace Calibration Test - Raw vs Calibrated")
        self.root.geometry("1100x700")
        self.root.configure(bg=COLORS['bg_dark'])
        self._setup_ui()

    def _setup_ui(self):
        main = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main.pack(fill='both', expand=True, padx=10, pady=10)

        # Top bar
        top = tk.Frame(main, bg=COLORS['bg_dark'])
        top.pack(fill='x', pady=(0, 10))
        tk.Label(top, text="DeepFace Calibration Test", font=('Helvetica', 18, 'bold'),
                 bg=COLORS['bg_dark'], fg=COLORS['text_white']).pack(side='left')
        self.user_label = tk.Label(top, text="[No User]", font=('Helvetica', 12),
                                   bg=COLORS['bg_dark'], fg=COLORS['text_gray'])
        self.user_label.pack(side='right')

        # Content
        content = tk.Frame(main, bg=COLORS['bg_dark'])
        content.pack(fill='both', expand=True, pady=10)

        # Left — camera
        left = tk.Frame(content, bg=COLORS['bg_dark'])
        left.pack(side='left', fill='both', expand=True)

        self.video_label = tk.Label(left, bg='#1a1a2e')
        self.video_label.pack(pady=10)

        self.status_label = tk.Label(left, text="Initializing...", font=('Helvetica', 11),
                                     bg=COLORS['bg_dark'], fg=COLORS['text_gray'])
        self.status_label.pack(pady=5)

        self.instruction_label = tk.Label(left, text="", font=('Helvetica', 12),
                                          bg=COLORS['bg_dark'], fg=COLORS['accent_yellow'],
                                          wraplength=400, justify='center')
        self.instruction_label.pack(pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(left, length=300, mode='determinate',
                                            variable=self.progress_var)
        self.progress_bar.pack(pady=5)
        self.progress_bar.pack_forget()

        # Right — comparison
        right = tk.Frame(content, bg=COLORS['bg_medium'], padx=15, pady=15)
        right.pack(side='right', fill='both', padx=(10, 0))

        tk.Label(right, text="Prediction Comparison", font=('Helvetica', 14, 'bold'),
                 bg=COLORS['bg_medium'], fg=COLORS['text_white']).pack(pady=(0, 15))

        comparison = tk.Frame(right, bg=COLORS['bg_medium'])
        comparison.pack(fill='both', expand=True)

        # Raw column
        raw_frame = tk.Frame(comparison, bg=COLORS['bg_dark'], padx=10, pady=10)
        raw_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))

        tk.Label(raw_frame, text="RAW OUTPUT", font=('Helvetica', 11, 'bold'),
                 bg=COLORS['bg_dark'], fg=COLORS['accent_red']).pack()

        self.raw_emotion = tk.Label(raw_frame, text="--", font=('Helvetica', 18, 'bold'),
                                    bg=COLORS['bg_dark'], fg=COLORS['text_white'])
        self.raw_emotion.pack(pady=10)

        self.raw_confidence = tk.Label(raw_frame, text="Confidence: --",
                                       font=('Helvetica', 10), bg=COLORS['bg_dark'], fg=COLORS['text_gray'])
        self.raw_confidence.pack()

        # Calibrated column
        cal_frame = tk.Frame(comparison, bg=COLORS['bg_dark'], padx=10, pady=10)
        cal_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))

        tk.Label(cal_frame, text="CALIBRATED OUTPUT", font=('Helvetica', 11, 'bold'),
                 bg=COLORS['bg_dark'], fg=COLORS['accent_green']).pack()

        self.cal_emotion = tk.Label(cal_frame, text="--", font=('Helvetica', 18, 'bold'),
                                    bg=COLORS['bg_dark'], fg=COLORS['text_white'])
        self.cal_emotion.pack(pady=10)

        self.cal_confidence = tk.Label(cal_frame, text="Confidence: --",
                                       font=('Helvetica', 10), bg=COLORS['bg_dark'], fg=COLORS['text_gray'])
        self.cal_confidence.pack()

        # Similarity bars
        sim_frame = tk.Frame(right, bg=COLORS['bg_medium'])
        sim_frame.pack(fill='x', pady=(15, 0))

        tk.Label(sim_frame, text="BASELINE SIMILARITIES", font=('Helvetica', 10, 'bold'),
                 bg=COLORS['bg_medium'], fg=COLORS['text_white']).pack(pady=(0, 8))

        self.sim_bars = {}
        for state, color in [('neutral', '#95A5A6'), ('happy', '#2ECC71'), ('calm', '#3498DB')]:
            row = tk.Frame(sim_frame, bg=COLORS['bg_medium'])
            row.pack(fill='x', pady=3)

            tk.Label(row, text=state.title(), font=('Helvetica', 10), width=8, anchor='e',
                     bg=COLORS['bg_medium'], fg=COLORS['text_gray']).pack(side='left')

            bar_bg = tk.Canvas(row, width=200, height=22, bg='#1e293b', highlightthickness=0)
            bar_bg.pack(side='left', padx=8)

            val_label = tk.Label(row, text="--", font=('Helvetica', 10, 'bold'), width=5,
                                 bg=COLORS['bg_medium'], fg=COLORS['text_white'])
            val_label.pack(side='left')

            self.sim_bars[state] = {'canvas': bar_bg, 'label': val_label, 'color': color}

        # Bottom bar
        bottom = tk.Frame(main, bg=COLORS['bg_dark'])
        bottom.pack(fill='x', pady=(10, 0))

        self.cal_btn = tk.Button(bottom, text="Calibrate", command=self.start_calibration,
                                 font=('Helvetica', 10), bg=COLORS['accent_blue'], fg='white', padx=15, pady=5)
        self.cal_btn.pack(side='left', padx=5)

        self.load_btn = tk.Button(bottom, text="Load Profile", command=self.load_profile,
                                  font=('Helvetica', 10), bg=COLORS['bg_medium'], fg='white', padx=15, pady=5)
        self.load_btn.pack(side='left', padx=5)

        self.save_btn = tk.Button(bottom, text="Save Profile", command=self.save_profile,
                                  font=('Helvetica', 10), bg=COLORS['bg_medium'], fg='white',
                                  padx=15, pady=5, state='disabled')
        self.save_btn.pack(side='left', padx=5)

        self.metrics_label = tk.Label(bottom, text="Latency: -- ms | FPS: --",
                                      font=('Helvetica', 10), bg=COLORS['bg_dark'], fg=COLORS['text_gray'])
        self.metrics_label.pack(side='right')

    # ========================================================================
    # Calibration
    # ========================================================================

    def start_calibration(self):
        user_id = simpledialog.askstring("User ID", "Enter user ID:", initialvalue="test_user")
        if not user_id:
            return
        self.current_user = user_id
        self.user_label.config(text=f"User: {user_id}")
        self.cal_in_progress = True
        self.cal_state_idx = 0
        self.captured_frames = []
        self.cal_btn.config(state='disabled')
        self.load_btn.config(state='disabled')
        self._start_state_capture()

    def _start_state_capture(self):
        if self.cal_state_idx >= len(CALIBRATION_STATES):
            self._complete_calibration()
            return
        state = CALIBRATION_STATES[self.cal_state_idx]
        self.instruction_label.config(
            text=f"Capturing {state['label'].upper()}\n\n{state['instruction']}")
        self.progress_bar.pack(pady=5)
        self.progress_var.set(0)
        self.captured_frames = []
        self.cal_start_time = time.time()
        self.status_label.config(text=f"Calibrating: {state['label']} ({state['duration']}s)")

    def _capture_frame_for_calibration(self, result: Dict):
        if not self.cal_in_progress:
            return
        state = CALIBRATION_STATES[self.cal_state_idx]
        elapsed = time.time() - self.cal_start_time
        self.progress_var.set(min(100, (elapsed / state['duration']) * 100))
        if elapsed > 1.0:
            self.captured_frames.append(result)
        if elapsed >= state['duration']:
            self._finalize_state_capture()

    def _finalize_state_capture(self):
        state = CALIBRATION_STATES[self.cal_state_idx]
        if len(self.captured_frames) < 5:
            messagebox.showwarning("Retry", f"Not enough frames for {state['label']}.")
            self._start_state_capture()
            return

        frames = self.captured_frames[-FRAMES_TO_AVERAGE:]
        avg_emb = average_embeddings([f['embedding'] for f in frames])

        if self.cal_state_idx == 0:
            self._temp_baseline = GenericBaseline(user_id=self.current_user, modality='deepface')

        self._temp_baseline.add_state(state['name'], avg_emb)

        self.cal_state_idx += 1
        self.captured_frames = []
        self.root.after(500, self._start_state_capture)

    def _complete_calibration(self):
        self.cal_in_progress = False
        self.progress_bar.pack_forget()
        self.instruction_label.config(text="")
        self.detector.set_baseline(self._temp_baseline)
        self.cal_btn.config(state='normal')
        self.load_btn.config(state='normal')
        self.save_btn.config(state='normal')
        self.status_label.config(text="Calibration complete! Now showing comparison.")
        messagebox.showinfo("Done", f"Calibration complete for '{self.current_user}'.")

    def save_profile(self):
        if self._temp_baseline is None:
            messagebox.showwarning("No Calibration", "Nothing to save.")
            return
        filepath = self.cal_manager.save_profile(self._temp_baseline)
        messagebox.showinfo("Saved", f"Profile saved to:\n{filepath}")

    def load_profile(self):
        profiles = self.cal_manager.list_profiles()
        if not profiles:
            messagebox.showinfo("No Profiles", "No saved profiles found.")
            return
        user_id = simpledialog.askstring("Load", f"Available: {', '.join(profiles)}\n\nEnter user ID:")
        if not user_id:
            return
        baseline = self.cal_manager.load_profile(user_id)
        if baseline is None:
            messagebox.showwarning("Not Found", f"Profile '{user_id}' not found.")
            return
        self.current_user = user_id
        self.user_label.config(text=f"User: {user_id}")
        self.detector.set_baseline(baseline)
        self._temp_baseline = baseline
        self.save_btn.config(state='normal')
        self.status_label.config(text=f"Loaded profile for '{user_id}'")

    # ========================================================================
    # Display
    # ========================================================================

    def update_video(self, frame, bbox=None):
        if frame is None:
            return
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (480, 360))
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def update_comparison(self, raw: Dict, calibrated: Dict):
        self.raw_emotion.config(text=raw['emotion'])
        self.raw_confidence.config(text=f"Confidence: {raw['confidence']:.0%}")

        if calibrated.get('calibrated'):
            source = calibrated.get('emotion_source', '')
            tag = {'calibration': '[CAL]', 'raw_model': '[RAW]'}.get(source, '')
            self.cal_emotion.config(text=calibrated['emotion'])
            self.cal_confidence.config(text=f"Confidence: {calibrated['confidence']:.0%} {tag}")

            sims = calibrated.get('similarities', {})
            closest = calibrated.get('closest_baseline', '')

            for state in ['neutral', 'happy', 'calm']:
                if state in sims:
                    sim = sims[state]
                    canvas = self.sim_bars[state]['canvas']
                    label = self.sim_bars[state]['label']
                    color = self.sim_bars[state]['color']

                    canvas.delete('all')
                    bar_width = max(0, min(200, int(200 * sim)))
                    if bar_width > 0:
                        fill = color if state == closest else '#475569'
                        canvas.create_rectangle(0, 0, bar_width, 22, fill=fill, outline='')

                    label.config(text=f"{sim:.3f}")
                    label.config(fg=color if state == closest else COLORS['text_gray'])
        else:
            self.cal_emotion.config(text="[No Cal]")
            self.cal_confidence.config(text="Calibrate first")
            for state in ['neutral', 'happy', 'calm']:
                self.sim_bars[state]['canvas'].delete('all')
                self.sim_bars[state]['label'].config(text="--")

    def show_no_face(self):
        self.raw_emotion.config(text="NO FACE")
        self.raw_confidence.config(text="--")
        self.cal_emotion.config(text="NO FACE")
        self.cal_confidence.config(text="--")

    # ========================================================================
    # Main Loop
    # ========================================================================

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

                face_img, bbox = self.face_detector.detect(frame)

                if face_img is not None:
                    start = time.time()
                    result = self.extractor.extract(face_img)
                    self.inference_time = time.time() - start

                    if self.cal_in_progress:
                        self.root.after(0, lambda r=result: self._capture_frame_for_calibration(r))

                    raw = self.detector.get_raw_prediction(result)
                    calibrated = self.detector.get_calibrated_prediction(result)

                    self.root.after(0, lambda f=frame.copy(), b=bbox: self.update_video(f, b))
                    self.root.after(0, lambda r=raw, c=calibrated: self.update_comparison(r, c))
                else:
                    self.root.after(0, lambda f=frame.copy(): self.update_video(f, None))
                    self.root.after(0, self.show_no_face)

                # FPS
                self.frame_count += 1
                elapsed = time.time() - self.fps_start_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.fps_start_time = time.time()
                    self.root.after(0, lambda: self.metrics_label.config(
                        text=f"Latency: {self.inference_time*1000:.0f}ms | FPS: {self.fps:.1f}"))

                time.sleep(0.01)

            except Exception as e:
                print(f"Loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        def init():
            self.status_label.config(text="Loading DeepFace model...")
            self.root.update()
            self.extractor.load(status_callback=lambda msg: self.root.after(
                0, lambda: self.status_label.config(text=msg)))

            self.status_label.config(text=f"Starting camera {self.camera_index}...")
            self.root.update()
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            time.sleep(0.5)

            self.running = True
            self.status_label.config(text="Ready! Click 'Calibrate' to start.")

            threading.Thread(target=self.main_loop, daemon=True).start()

        threading.Thread(target=init, daemon=True).start()
        self.root.mainloop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='DeepFace Calibration Demo')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (0=default, 1=MacBook webcam)')
    args = parser.parse_args()

    print(f"Using camera index: {args.camera}")
    app = DeepFaceDemoApp(camera_index=args.camera)
    app.run()
