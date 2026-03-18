"""
Calibration Comparison Demo: Embedding vs Action Unit Approaches

Shows cosine similarity scores side-by-side for both calibration methods
on the same camera feed.

Usage:
    python calibration_comparison_demo.py --camera 1
"""

import time
import threading
import math
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
from typing import Optional, Dict, List

import mediapipe as mp
from calibration_core import HSEmotionExtractor, average_embeddings, average_values


# ============================================================================
# Configuration
# ============================================================================

FRAMES_TO_AVERAGE = 25

CAL_STATES = [
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
    'accent_purple': '#9B59B6',
    'accent_orange': '#E67E22',
}


# ============================================================================
# Face Detector (OpenCV Haar Cascade for embedding approach)
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
# Action Unit Extractor (from Sourick's approach)
# ============================================================================

class MediaPipeExtractor:
    """Extract both AU vector and raw landmarks from MediaPipe Face Landmarker."""

    def __init__(self, model_path: str = None):
        if model_path is None:
            import os
            model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            output_face_blendshapes=False,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def _distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def extract(self, frame_rgb) -> Optional[Dict]:
        """
        Extract AU vector (8-dim) and raw landmarks (1434-dim) from an RGB frame.

        Returns:
            Dict with 'au' (8-dim) and 'landmarks' (1434-dim), or None
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.landmarker.detect(mp_image)

        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return None

        lm = results.face_landmarks[0]

        # Reference distances for normalization
        face_width = self._distance(lm[234], lm[454])
        face_height = self._distance(lm[10], lm[152])

        if face_width == 0 or face_height == 0:
            return None

        # --- Action Units (8-dim) ---
        AU12 = self._distance(lm[61], lm[291]) / face_width
        AU15 = self._distance(lm[61], lm[291]) / face_width
        AU6 = ((self._distance(lm[159], lm[33]) +
                self._distance(lm[386], lm[263])) / 2) / face_height
        AU5 = self._distance(lm[159], lm[386]) / face_height
        AU1 = ((self._distance(lm[70], lm[63]) +
                self._distance(lm[105], lm[334])) / 2) / face_height
        AU2 = self._distance(lm[55], lm[285]) / face_height
        AU4 = ((self._distance(lm[70], lm[105]) +
                self._distance(lm[63], lm[334])) / 2) / face_height
        AU9 = self._distance(lm[195], lm[5]) / face_height

        au_vec = np.array([AU12, AU6, AU1, AU4, AU15, AU2, AU5, AU9])

        # --- Raw Landmarks (478 x 3 = 1434-dim) ---
        # Normalize by face width/height for scale invariance
        landmarks_flat = []
        for point in lm:
            landmarks_flat.extend([point.x / face_width, point.y / face_height, point.z / face_width])
        landmarks_vec = np.array(landmarks_flat)

        return {'au': au_vec, 'landmarks': landmarks_vec}


# ============================================================================
# Cosine Similarity
# ============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ============================================================================
# Main Application
# ============================================================================

class CalibrationComparisonApp:
    """Side-by-side comparison of embedding vs action unit calibration."""

    def __init__(self, camera_index: int = 0):
        # Extractors
        self.emb_extractor = HSEmotionExtractor()
        self.mp_extractor = MediaPipeExtractor()
        self.face_detector = FaceDetector()

        # Camera
        self.camera_index = camera_index
        self.cap = None

        # Baselines for each approach
        self.emb_baselines: Dict[str, np.ndarray] = {}   # 1280-dim
        self.au_baselines: Dict[str, np.ndarray] = {}     # 8-dim
        self.lm_baselines: Dict[str, np.ndarray] = {}     # 1434-dim

        # State
        self.running = False
        self.current_user: Optional[str] = None
        self.cal_in_progress = False
        self.cal_state_idx = 0
        self.cal_start_time = 0.0
        self.cal_captured_emb: List[np.ndarray] = []
        self.cal_captured_au: List[np.ndarray] = []
        self.cal_captured_lm: List[np.ndarray] = []

        # Latest results (thread-safe)
        self._lock = threading.Lock()
        self._latest_emb_sims: Optional[Dict[str, float]] = None
        self._latest_au_sims: Optional[Dict[str, float]] = None
        self._latest_lm_sims: Optional[Dict[str, float]] = None
        self._latest_emb_emotion: str = "--"
        self._latest_au_emotion: str = "--"
        self._latest_lm_emotion: str = "--"
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_bbox: Optional[tuple] = None
        self._face_detected = False

        # Metrics
        self.emb_latency = 0.0
        self.au_latency = 0.0

        # GUI
        self.root = tk.Tk()
        self.root.title("Calibration Comparison: Embeddings vs Action Units")
        self.root.geometry("1450x700")
        self.root.configure(bg=COLORS['bg_dark'])
        self._setup_ui()

    # ========================================================================
    # UI
    # ========================================================================

    def _setup_ui(self):
        main = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main.pack(fill='both', expand=True, padx=10, pady=10)

        # Top bar
        top = tk.Frame(main, bg=COLORS['bg_dark'])
        top.pack(fill='x', pady=(0, 5))
        tk.Label(top, text="Calibration Comparison", font=('Helvetica', 18, 'bold'),
                 bg=COLORS['bg_dark'], fg=COLORS['text_white']).pack(side='left')
        self.user_label = tk.Label(top, text="[No User]", font=('Helvetica', 12),
                                   bg=COLORS['bg_dark'], fg=COLORS['text_gray'])
        self.user_label.pack(side='right')

        # Content
        content = tk.Frame(main, bg=COLORS['bg_dark'])
        content.pack(fill='both', expand=True, pady=10)

        # Left — camera
        left = tk.Frame(content, bg=COLORS['bg_dark'], width=500)
        left.pack(side='left', fill='both')
        left.pack_propagate(False)

        self.video_label = tk.Label(left, bg='#1a1a2e')
        self.video_label.pack(pady=10)

        self.status_label = tk.Label(left, text="Initializing...", font=('Helvetica', 11),
                                     bg=COLORS['bg_dark'], fg=COLORS['text_gray'])
        self.status_label.pack(pady=5)

        self.instruction_label = tk.Label(left, text="", font=('Helvetica', 11),
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
        right.pack(side='right', fill='both', expand=True, padx=(10, 0))

        tk.Label(right, text="Cosine Similarity Scores", font=('Helvetica', 14, 'bold'),
                 bg=COLORS['bg_medium'], fg=COLORS['text_white']).pack(pady=(0, 15))

        # Two columns
        cols = tk.Frame(right, bg=COLORS['bg_medium'])
        cols.pack(fill='both', expand=True)

        self.emb_widgets = self._create_sim_column(cols, "EMBEDDING", "(1280-dim, HSEmotion)", COLORS['accent_blue'])
        self.lm_widgets = self._create_sim_column(cols, "LANDMARKS", "(1434-dim, MediaPipe)", COLORS['accent_purple'])
        self.au_widgets = self._create_sim_column(cols, "ACTION UNITS", "(8-dim, MediaPipe)", COLORS['accent_orange'])

        # Agreement indicator
        self.agreement_label = tk.Label(right, text="", font=('Helvetica', 12, 'bold'),
                                         bg=COLORS['bg_medium'], fg=COLORS['text_gray'])
        self.agreement_label.pack(pady=(10, 0))

        # Bottom bar
        bottom = tk.Frame(main, bg=COLORS['bg_dark'])
        bottom.pack(fill='x', pady=(10, 0))

        self.cal_btn = tk.Button(bottom, text="Calibrate Both", command=self.start_calibration,
                                 font=('Helvetica', 10), bg=COLORS['accent_blue'], fg='white', padx=15, pady=5)
        self.cal_btn.pack(side='left', padx=5)

        self.metrics_label = tk.Label(bottom, text="EMB: -- ms | MediaPipe: -- ms",
                                      font=('Helvetica', 10), bg=COLORS['bg_dark'], fg=COLORS['text_gray'])
        self.metrics_label.pack(side='right')

    def _create_sim_column(self, parent, title: str, subtitle: str, color: str) -> Dict:
        frame = tk.Frame(parent, bg=COLORS['bg_dark'], padx=12, pady=12)
        frame.pack(side='left', fill='both', expand=True, padx=5)

        tk.Label(frame, text=title, font=('Helvetica', 11, 'bold'),
                 bg=COLORS['bg_dark'], fg=color).pack()
        tk.Label(frame, text=subtitle, font=('Helvetica', 9),
                 bg=COLORS['bg_dark'], fg=COLORS['text_gray']).pack(pady=(0, 10))

        # Similarity bars
        bars = {}
        for state, bar_color in [('neutral', '#95A5A6'), ('happy', '#2ECC71'), ('calm', '#3498DB')]:
            row = tk.Frame(frame, bg=COLORS['bg_dark'])
            row.pack(fill='x', pady=4)

            tk.Label(row, text=state.title(), font=('Helvetica', 10), width=8, anchor='e',
                     bg=COLORS['bg_dark'], fg=COLORS['text_gray']).pack(side='left')

            bar_bg = tk.Canvas(row, width=180, height=22, bg='#1e293b', highlightthickness=0)
            bar_bg.pack(side='left', padx=8)

            val_label = tk.Label(row, text="--", font=('Helvetica', 10, 'bold'), width=5,
                                 bg=COLORS['bg_dark'], fg=COLORS['text_white'])
            val_label.pack(side='left')

            bars[state] = {'canvas': bar_bg, 'label': val_label, 'color': bar_color}

        # Separator
        tk.Frame(frame, bg='#475569', height=1).pack(fill='x', pady=10)

        # Detected emotion
        emotion_label = tk.Label(frame, text="--", font=('Helvetica', 18, 'bold'),
                                 bg=COLORS['bg_dark'], fg=COLORS['text_white'])
        emotion_label.pack(pady=5)

        return {'bars': bars, 'emotion': emotion_label}

    # ========================================================================
    # Calibration
    # ========================================================================

    def start_calibration(self):
        user_id = simpledialog.askstring("User ID", "Enter user ID:", initialvalue="test_user")
        if not user_id:
            return

        self.current_user = user_id
        self.user_label.config(text=f"User: {user_id}")

        self.emb_baselines = {}
        self.au_baselines = {}
        self.lm_baselines = {}
        self.cal_in_progress = True
        self.cal_state_idx = 0
        self.cal_btn.config(state='disabled')

        self._start_cal_state()

    def _start_cal_state(self):
        if self.cal_state_idx >= len(CAL_STATES):
            self._complete_calibration()
            return

        state = CAL_STATES[self.cal_state_idx]
        self.instruction_label.config(
            text=f"Capturing {state['label'].upper()}\n\n{state['instruction']}"
        )
        self.progress_bar.pack(pady=5)
        self.progress_var.set(0)
        self.cal_captured_emb = []
        self.cal_captured_au = []
        self.cal_captured_lm = []
        self.cal_start_time = time.time()
        self.status_label.config(text=f"Calibrating: {state['label']} ({state['duration']}s)")

    def _capture_cal_frame(self, emb: np.ndarray, au: Optional[np.ndarray], lm: Optional[np.ndarray]):
        if not self.cal_in_progress:
            return

        elapsed = time.time() - self.cal_start_time
        state = CAL_STATES[self.cal_state_idx]

        self.progress_var.set(min(100, (elapsed / state['duration']) * 100))

        # Skip first second
        if elapsed > 1.0:
            self.cal_captured_emb.append(emb)
            if au is not None:
                self.cal_captured_au.append(au)
            if lm is not None:
                self.cal_captured_lm.append(lm)

        if elapsed >= state['duration']:
            self._finalize_cal_state()

    def _finalize_cal_state(self):
        state = CAL_STATES[self.cal_state_idx]

        if len(self.cal_captured_emb) < 5 or len(self.cal_captured_au) < 5:
            messagebox.showwarning("Retry", f"Not enough frames for {state['label']}.")
            self._start_cal_state()
            return

        # Average embeddings (1280-dim)
        emb_frames = self.cal_captured_emb[-FRAMES_TO_AVERAGE:]
        self.emb_baselines[state['name']] = average_embeddings(emb_frames)

        # Average AUs (8-dim)
        au_frames = self.cal_captured_au[-FRAMES_TO_AVERAGE:]
        self.au_baselines[state['name']] = np.mean(np.stack(au_frames), axis=0)

        # Average landmarks (1434-dim)
        if len(self.cal_captured_lm) >= 5:
            lm_frames = self.cal_captured_lm[-FRAMES_TO_AVERAGE:]
            self.lm_baselines[state['name']] = np.mean(np.stack(lm_frames), axis=0)

        self.cal_state_idx += 1
        self.cal_captured_emb = []
        self.cal_captured_au = []
        self.cal_captured_lm = []
        self.root.after(500, self._start_cal_state)

    def _complete_calibration(self):
        self.cal_in_progress = False
        self.progress_bar.pack_forget()
        self.instruction_label.config(text="")
        self.cal_btn.config(state='normal')
        self.status_label.config(text="Calibration complete! Now comparing both approaches.")
        messagebox.showinfo("Done", "Calibration complete for both approaches!")

    # ========================================================================
    # Display
    # ========================================================================

    def _update_sim_bars(self, widgets: Dict, sims: Optional[Dict[str, float]], emotion: str):
        if sims is None:
            for state in ['neutral', 'happy', 'calm']:
                widgets['bars'][state]['canvas'].delete('all')
                widgets['bars'][state]['label'].config(text="--")
            widgets['emotion'].config(text="--")
            return

        closest = max(sims, key=sims.get)

        for state in ['neutral', 'happy', 'calm']:
            sim = sims[state]
            canvas = widgets['bars'][state]['canvas']
            color = widgets['bars'][state]['color']
            label = widgets['bars'][state]['label']

            canvas.delete('all')
            bar_width = max(0, min(180, int(180 * sim)))
            if bar_width > 0:
                fill_color = color if state == closest else '#475569'
                canvas.create_rectangle(0, 0, bar_width, 22, fill=fill_color, outline='')

            label.config(text=f"{sim:.2f}")
            if state == closest:
                label.config(fg=color)
            else:
                label.config(fg=COLORS['text_gray'])

        widgets['emotion'].config(text=emotion)

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

    def update_display(self):
        """GUI timer — reads latest results and updates everything."""
        if not self.running:
            return

        self.root.after(100, self.update_display)

        with self._lock:
            emb_sims = self._latest_emb_sims.copy() if self._latest_emb_sims else None
            au_sims = self._latest_au_sims.copy() if self._latest_au_sims else None
            lm_sims = self._latest_lm_sims.copy() if self._latest_lm_sims else None
            emb_emotion = self._latest_emb_emotion
            au_emotion = self._latest_au_emotion
            lm_emotion = self._latest_lm_emotion
            frame = self._latest_frame.copy() if self._latest_frame is not None else None
            bbox = self._latest_bbox
            face_ok = self._face_detected

        if frame is not None:
            self.update_video(frame, bbox)

        if not face_ok:
            self._update_sim_bars(self.emb_widgets, None, "NO FACE")
            self._update_sim_bars(self.lm_widgets, None, "NO FACE")
            self._update_sim_bars(self.au_widgets, None, "NO FACE")
            self.agreement_label.config(text="", fg=COLORS['text_gray'])
            return

        calibrated = len(self.emb_baselines) == 3 and len(self.au_baselines) == 3

        if calibrated:
            self._update_sim_bars(self.emb_widgets, emb_sims, emb_emotion)
            self._update_sim_bars(self.lm_widgets, lm_sims, lm_emotion)
            self._update_sim_bars(self.au_widgets, au_sims, au_emotion)

            # Agreement check
            emotions = [emb_emotion, lm_emotion, au_emotion]
            if emb_emotion == lm_emotion == au_emotion:
                self.agreement_label.config(
                    text=f"All agree: {emb_emotion}",
                    fg=COLORS['accent_green']
                )
            else:
                self.agreement_label.config(
                    text=f"EMB={emb_emotion} | LM={lm_emotion} | AU={au_emotion}",
                    fg=COLORS['accent_red']
                )
        else:
            self._update_sim_bars(self.emb_widgets, None, "Calibrate first")
            self._update_sim_bars(self.lm_widgets, None, "Calibrate first")
            self._update_sim_bars(self.au_widgets, None, "Calibrate first")

        self.metrics_label.config(
            text=f"EMB: {self.emb_latency*1000:.0f}ms | MediaPipe: {self.au_latency*1000:.0f}ms"
        )

    # ========================================================================
    # Processing Thread
    # ========================================================================

    def process_loop(self):
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(0.1)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Face detection for embedding approach
                face_img, bbox = self.face_detector.detect(frame)

                # MediaPipe extraction (AU + landmarks from same model)
                au_start = time.time()
                mp_result = self.mp_extractor.extract(frame_rgb)
                self.au_latency = time.time() - au_start

                au_vec = mp_result['au'] if mp_result else None
                lm_vec = mp_result['landmarks'] if mp_result else None

                if face_img is not None and au_vec is not None:
                    # Embedding extraction
                    emb_start = time.time()
                    result = self.emb_extractor.extract(face_img)
                    self.emb_latency = time.time() - emb_start

                    emb = result['embedding']

                    # Calibration capture
                    if self.cal_in_progress:
                        self.root.after(0, lambda e=emb, a=au_vec, l=lm_vec: self._capture_cal_frame(e, a, l))

                    # Compute similarities if calibrated
                    emb_sims = None
                    au_sims = None
                    lm_sims = None
                    emb_emotion = "--"
                    au_emotion = "--"
                    lm_emotion = "--"

                    if len(self.emb_baselines) == 3:
                        emb_sims = {
                            state: cosine_similarity(emb, baseline)
                            for state, baseline in self.emb_baselines.items()
                        }
                        emb_emotion = max(emb_sims, key=emb_sims.get).title()

                    if len(self.au_baselines) == 3:
                        au_sims = {
                            state: cosine_similarity(au_vec, baseline)
                            for state, baseline in self.au_baselines.items()
                        }
                        au_emotion = max(au_sims, key=au_sims.get).title()

                    if len(self.lm_baselines) == 3 and lm_vec is not None:
                        lm_sims = {
                            state: cosine_similarity(lm_vec, baseline)
                            for state, baseline in self.lm_baselines.items()
                        }
                        lm_emotion = max(lm_sims, key=lm_sims.get).title()

                    with self._lock:
                        self._latest_frame = frame.copy()
                        self._latest_bbox = bbox
                        self._latest_emb_sims = emb_sims
                        self._latest_au_sims = au_sims
                        self._latest_lm_sims = lm_sims
                        self._latest_emb_emotion = emb_emotion
                        self._latest_au_emotion = au_emotion
                        self._latest_lm_emotion = lm_emotion
                        self._face_detected = True
                else:
                    with self._lock:
                        self._latest_frame = frame.copy()
                        self._latest_bbox = None
                        self._face_detected = False

                time.sleep(0.01)

            except Exception as e:
                print(f"Process loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    # ========================================================================
    # Lifecycle
    # ========================================================================

    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        def init():
            self.status_label.config(text="Loading HSEmotion model...")
            self.root.update()
            self.emb_extractor.load()

            self.status_label.config(text=f"Starting camera {self.camera_index}...")
            self.root.update()
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            time.sleep(0.5)

            self.running = True
            self.status_label.config(text="Ready! Click 'Calibrate Both' to start.")

            threading.Thread(target=self.process_loop, daemon=True).start()
            self.root.after(100, self.update_display)

        threading.Thread(target=init, daemon=True).start()
        self.root.mainloop()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Calibration Comparison: EMB vs AU')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (0=default, 1=MacBook webcam)')
    args = parser.parse_args()

    app = CalibrationComparisonApp(camera_index=args.camera)
    app.run()
