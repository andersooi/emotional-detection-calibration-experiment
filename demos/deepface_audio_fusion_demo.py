"""
DeepFace + Emotion2Vec Fusion Demo

DeepFace (calibrated neutral + happy) as the visual branch.
Emotion2Vec large (raw, no calibration) as the audio branch.
V1 Probability Fusion with asymmetric sadness weighting.

The fusion adapter ensures calibration affects the fused output
by modifying the probability vector, not just the label.

Usage:
    python demos/deepface_audio_fusion_demo.py --camera 1
"""

import time
import threading
import queue
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
from typing import Optional, Dict, List

from core import (
    DeepFaceEmotionEmbeddingExtractor,
    Emotion2VecExtractor,
    GenericBaseline,
    GenericCalibrationManager,
    GenericCalibratedDetector,
    average_embeddings,
    cosine_similarity,
    ProbabilityFusion,
    FusionResult,
    QUADRANT_LABELS,
    build_face_result,
)

try:
    import sounddevice as sd
except ImportError:
    sd = None


# ============================================================================
# Configuration
# ============================================================================

SAMPLE_RATE = 16000
CHUNK_DURATION = 2.0  # Reduced from 3s for faster audio updates (~1s between results)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
FRAMES_TO_AVERAGE = 25
AUDIO_STALE_THRESHOLD = 4.0  # Reduced with shorter chunks (was 5s with 3s chunks)


CALIBRATION_STATES = [
    {'name': 'neutral', 'label': 'Neutral', 'duration': 5,
     'instruction': 'Look at the camera with a relaxed, natural expression.\n'
                    'This anchors YOUR resting face.'},
    {'name': 'happy', 'label': 'Happy', 'duration': 5,
     'instruction': 'Think of a happy memory. Let yourself smile naturally.'},
]

EMOTION_LABELS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

EMOTION_COLORS = {
    'Anger': '#E74C3C', 'Disgust': '#8E44AD', 'Fear': '#9B59B6',
    'Happiness': '#F1C40F', 'Sadness': '#3498DB', 'Surprise': '#E67E22',
    'Neutral': '#95A5A6',
}

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
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return None, None
        areas = [w * h for (x, y, w, h) in faces]
        idx = np.argmax(areas)
        x, y, w, h = faces[idx]
        margin = int(0.1 * w)
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)
        face_bgr = frame[y1:y2, x1:x2].copy()
        return face_bgr, (x1, y1, x2, y2)


# ============================================================================
# Audio Capture
# ============================================================================

class AudioCapture:
    def __init__(self, sample_rate=SAMPLE_RATE, chunk_duration=CHUNK_DURATION):
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.running = False
        self.audio_queue = queue.Queue()
        self.buffer = np.array([], dtype=np.float32)
        self.stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        audio_chunk = indata[:, 0].copy()
        self.buffer = np.concatenate([self.buffer, audio_chunk])
        while len(self.buffer) >= self.chunk_samples:
            chunk = self.buffer[:self.chunk_samples]
            self.buffer = self.buffer[self.chunk_samples // 4:]  # 75% overlap
            self.audio_queue.put(chunk)

    def start(self):
        if sd is None:
            raise RuntimeError("sounddevice not installed")
        self.running = True
        self.buffer = np.array([], dtype=np.float32)
        self.stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype=np.float32,
            callback=self._audio_callback, blocksize=int(self.sample_rate * 0.1))
        self.stream.start()

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_chunk(self, timeout=0.1):
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ============================================================================
# Main Application
# ============================================================================

class DeepFaceAudioFusionApp:
    """DeepFace (calibrated) + Emotion2Vec (raw) fusion demo."""

    def __init__(self, camera_index: int = 0):
        # Models
        self.face_extractor = DeepFaceEmotionEmbeddingExtractor()
        self.audio_extractor = Emotion2VecExtractor(model_size='large')
        self.face_detector = FaceDetector()
        self.audio_capture = AudioCapture()

        # Calibration (face only)
        self.cal_manager = GenericCalibrationManager(modality='deepface_emb')
        self.cal_detector = GenericCalibratedDetector(
            calibrated_emotions={'Happiness', 'Neutral'})

        # Fusion (V1 only)
        self.fusion = ProbabilityFusion()

        # Camera
        self.camera_index = camera_index
        self.cap = None
        self.running = False

        # Thread-safe results
        self._lock = threading.Lock()
        self._latest_face_for_fusion: Optional[Dict] = None
        self._latest_face_raw: Optional[Dict] = None
        self._latest_face_cal: Optional[Dict] = None
        self._latest_audio_result: Optional[Dict] = None
        self._face_timestamp = 0.0
        self._audio_timestamp = 0.0

        # Calibration state
        self.cal_in_progress = False
        self.cal_state_idx = 0
        self.cal_start_time = 0.0
        self.captured_frames: List[Dict] = []
        self._temp_baseline: Optional[GenericBaseline] = None
        self.current_user: Optional[str] = None

        # No-face grace
        self.no_face_count = 0
        self.no_face_grace = 3

        # Face EMA smoothing (dampen jitter from mouth movement during speech)
        self.face_probs_ema: Optional[np.ndarray] = None
        self.face_ema_decay = 0.3  # weight of new frame (lower = smoother)

        # Metrics
        self.face_latency = 0.0
        self.face_fps = 0.0
        self._face_frame_count = 0
        self._face_fps_time = time.time()

        # GUI
        self.root = tk.Tk()
        self.root.title("DeepFace + Emotion2Vec Fusion")
        self.root.geometry("1350x1100")
        self.root.configure(bg=COLORS['bg_dark'])
        self._setup_ui()

    # ========================================================================
    # UI Setup
    # ========================================================================

    def _setup_ui(self):
        main = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main.pack(fill='both', expand=True, padx=10, pady=10)

        # Top bar
        top = tk.Frame(main, bg=COLORS['bg_dark'])
        top.pack(fill='x', pady=(0, 10))
        tk.Label(top, text="DeepFace + Emotion2Vec Fusion",
                 font=('Helvetica', 18, 'bold'),
                 bg=COLORS['bg_dark'], fg=COLORS['text_white']).pack(side='left')
        self.user_label = tk.Label(top, text="[No User]",
                                   font=('Helvetica', 12),
                                   bg=COLORS['bg_dark'], fg=COLORS['text_gray'])
        self.user_label.pack(side='right')

        # Content: camera left, panels right
        content = tk.Frame(main, bg=COLORS['bg_dark'])
        content.pack(fill='both', expand=True)

        # Left — camera + status
        left = tk.Frame(content, bg=COLORS['bg_dark'], width=500)
        left.pack(side='left', fill='both')

        self.video_label = tk.Label(left, bg='#1a1a2e')
        self.video_label.pack(pady=10)

        self.status_label = tk.Label(left, text="Initializing...",
                                     font=('Helvetica', 11),
                                     bg=COLORS['bg_dark'], fg=COLORS['text_gray'])
        self.status_label.pack(pady=5)

        self.instruction_label = tk.Label(left, text="",
                                          font=('Helvetica', 12),
                                          bg=COLORS['bg_dark'],
                                          fg=COLORS['accent_yellow'],
                                          wraplength=400, justify='center')
        self.instruction_label.pack(pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(left, length=300,
                                            mode='determinate',
                                            variable=self.progress_var)
        self.progress_bar.pack(pady=5)
        self.progress_bar.pack_forget()

        # Right — three columns: Face | Audio | Fused
        right = tk.Frame(content, bg=COLORS['bg_medium'], padx=10, pady=10)
        right.pack(side='right', fill='both', expand=True, padx=(10, 0))

        # Column headers
        cols = tk.Frame(right, bg=COLORS['bg_medium'])
        cols.pack(fill='x', pady=(0, 5))

        # Face column
        face_col = tk.Frame(cols, bg=COLORS['bg_dark'], padx=8, pady=8)
        face_col.pack(side='left', fill='both', expand=True, padx=(0, 3))
        tk.Label(face_col, text="FACE (DeepFace)",
                 font=('Helvetica', 10, 'bold'),
                 bg=COLORS['bg_dark'], fg=COLORS['accent_red']).pack()
        self.face_emotion_lbl = tk.Label(face_col, text="--",
                                         font=('Helvetica', 16, 'bold'),
                                         bg=COLORS['bg_dark'],
                                         fg=COLORS['text_white'])
        self.face_emotion_lbl.pack(pady=5)
        self.face_info_lbl = tk.Label(face_col, text="--",
                                      font=('Helvetica', 9),
                                      bg=COLORS['bg_dark'],
                                      fg=COLORS['text_gray'])
        self.face_info_lbl.pack()

        # Audio column
        audio_col = tk.Frame(cols, bg=COLORS['bg_dark'], padx=8, pady=8)
        audio_col.pack(side='left', fill='both', expand=True, padx=3)
        tk.Label(audio_col, text="AUDIO (Emotion2Vec)",
                 font=('Helvetica', 10, 'bold'),
                 bg=COLORS['bg_dark'], fg=COLORS['accent_purple']).pack()
        self.audio_emotion_lbl = tk.Label(audio_col, text="--",
                                          font=('Helvetica', 16, 'bold'),
                                          bg=COLORS['bg_dark'],
                                          fg=COLORS['text_white'])
        self.audio_emotion_lbl.pack(pady=5)
        self.audio_info_lbl = tk.Label(audio_col, text="--",
                                       font=('Helvetica', 9),
                                       bg=COLORS['bg_dark'],
                                       fg=COLORS['text_gray'])
        self.audio_info_lbl.pack()

        # Fused column
        fused_col = tk.Frame(cols, bg=COLORS['bg_dark'], padx=8, pady=8)
        fused_col.pack(side='left', fill='both', expand=True, padx=(3, 0))
        tk.Label(fused_col, text="FUSED OUTPUT",
                 font=('Helvetica', 10, 'bold'),
                 bg=COLORS['bg_dark'], fg=COLORS['accent_green']).pack()
        self.fused_emotion_lbl = tk.Label(fused_col, text="--",
                                          font=('Helvetica', 16, 'bold'),
                                          bg=COLORS['bg_dark'],
                                          fg=COLORS['text_white'])
        self.fused_emotion_lbl.pack(pady=5)
        self.fused_info_lbl = tk.Label(fused_col, text="--",
                                       font=('Helvetica', 9),
                                       bg=COLORS['bg_dark'],
                                       fg=COLORS['text_gray'])
        self.fused_info_lbl.pack()

        # Diagnostics section
        diag = tk.Frame(right, bg=COLORS['bg_medium'])
        diag.pack(fill='x', pady=(8, 5))

        tk.Label(diag, text="FUSION WEIGHTS",
                 font=('Helvetica', 10, 'bold'),
                 bg=COLORS['bg_medium'], fg=COLORS['text_white']).pack(pady=(0, 3))
        self.weights_lbl = tk.Label(diag, text="Face: --% | Audio: --%",
                                    font=('Courier', 10),
                                    bg=COLORS['bg_medium'],
                                    fg=COLORS['text_gray'])
        self.weights_lbl.pack()
        self.adapter_lbl = tk.Label(diag, text="Adapter: --",
                                    font=('Courier', 9),
                                    bg=COLORS['bg_medium'],
                                    fg=COLORS['text_gray'])
        self.adapter_lbl.pack()

        # Similarity bars
        sim_frame = tk.Frame(right, bg=COLORS['bg_medium'])
        sim_frame.pack(fill='x', pady=(5, 5))
        tk.Label(sim_frame, text="BASELINE SIMILARITIES",
                 font=('Helvetica', 10, 'bold'),
                 bg=COLORS['bg_medium'], fg=COLORS['text_white']).pack(pady=(0, 4))

        self.sim_bars = {}
        for state, color in [('neutral', '#95A5A6'), ('happy', '#2ECC71')]:
            row = tk.Frame(sim_frame, bg=COLORS['bg_medium'])
            row.pack(fill='x', pady=2)
            tk.Label(row, text=state.title(), font=('Helvetica', 9),
                     width=8, anchor='e',
                     bg=COLORS['bg_medium'],
                     fg=COLORS['text_gray']).pack(side='left')
            canvas = tk.Canvas(row, width=160, height=16,
                               bg='#1e293b', highlightthickness=0)
            canvas.pack(side='left', padx=8)
            val_lbl = tk.Label(row, text="--",
                               font=('Helvetica', 9, 'bold'), width=5,
                               bg=COLORS['bg_medium'],
                               fg=COLORS['text_white'])
            val_lbl.pack(side='left')
            self.sim_bars[state] = {'canvas': canvas, 'label': val_lbl, 'color': color}

        # Face probability bars — raw (before calibration adapter)
        raw_prob_frame = tk.Frame(right, bg=COLORS['bg_medium'])
        raw_prob_frame.pack(fill='x', pady=(5, 0))
        tk.Label(raw_prob_frame, text="FACE RAW",
                 font=('Helvetica', 10, 'bold'),
                 bg=COLORS['bg_medium'], fg=COLORS['accent_red']).pack(pady=(0, 2))

        self.raw_prob_bars = {}
        for label in EMOTION_LABELS:
            row = tk.Frame(raw_prob_frame, bg=COLORS['bg_medium'])
            row.pack(fill='x', pady=1)
            tk.Label(row, text=label[:8], font=('Helvetica', 8),
                     width=10, anchor='e',
                     bg=COLORS['bg_medium'],
                     fg=COLORS['text_gray']).pack(side='left')
            canvas = tk.Canvas(row, width=100, height=12,
                               bg='#1e293b', highlightthickness=0)
            canvas.pack(side='left', padx=(6, 2))
            val_lbl = tk.Label(row, text="--", font=('Helvetica', 7),
                               width=5, bg=COLORS['bg_medium'],
                               fg=COLORS['text_gray'])
            val_lbl.pack(side='left')
            self.raw_prob_bars[label] = {'canvas': canvas, 'val': val_lbl,
                                         'color': EMOTION_COLORS[label]}

        # Face probability bars — adapted (after calibration adapter, what fusion sees)
        prob_frame = tk.Frame(right, bg=COLORS['bg_medium'])
        prob_frame.pack(fill='both', expand=True, pady=(5, 0))
        tk.Label(prob_frame, text="FACE ADAPTED (into fusion)",
                 font=('Helvetica', 10, 'bold'),
                 bg=COLORS['bg_medium'], fg=COLORS['accent_green']).pack(pady=(0, 2))

        self.prob_bars = {}
        for label in EMOTION_LABELS:
            row = tk.Frame(prob_frame, bg=COLORS['bg_medium'])
            row.pack(fill='x', pady=1)
            tk.Label(row, text=label[:8], font=('Helvetica', 9),
                     width=10, anchor='e',
                     bg=COLORS['bg_medium'],
                     fg=COLORS['text_gray']).pack(side='left')
            canvas = tk.Canvas(row, width=140, height=14,
                               bg='#1e293b', highlightthickness=0)
            canvas.pack(side='left', padx=(8, 2))
            val_lbl = tk.Label(row, text="--", font=('Helvetica', 8),
                               width=5, bg=COLORS['bg_medium'],
                               fg=COLORS['text_gray'])
            val_lbl.pack(side='left')
            self.prob_bars[label] = {'canvas': canvas, 'val': val_lbl,
                                     'color': EMOTION_COLORS[label]}

        # Audio probability bars
        audio_prob_frame = tk.Frame(right, bg=COLORS['bg_medium'])
        audio_prob_frame.pack(fill='both', expand=True, pady=(5, 0))
        tk.Label(audio_prob_frame, text="AUDIO PROBABILITIES (raw)",
                 font=('Helvetica', 10, 'bold'),
                 bg=COLORS['bg_medium'], fg=COLORS['text_white']).pack(pady=(0, 4))

        audio_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        audio_colors = {
            'Angry': '#E74C3C', 'Disgust': '#8E44AD', 'Fear': '#9B59B6',
            'Happy': '#F1C40F', 'Neutral': '#95A5A6', 'Sad': '#3498DB',
            'Surprise': '#E67E22',
        }
        self.audio_prob_bars = {}
        for label in audio_labels:
            row = tk.Frame(audio_prob_frame, bg=COLORS['bg_medium'])
            row.pack(fill='x', pady=1)
            tk.Label(row, text=label, font=('Helvetica', 9),
                     width=10, anchor='e',
                     bg=COLORS['bg_medium'],
                     fg=COLORS['text_gray']).pack(side='left')
            canvas = tk.Canvas(row, width=140, height=14,
                               bg='#1e293b', highlightthickness=0)
            canvas.pack(side='left', padx=(8, 2))
            val_lbl = tk.Label(row, text="--", font=('Helvetica', 8),
                               width=5, bg=COLORS['bg_medium'],
                               fg=COLORS['text_gray'])
            val_lbl.pack(side='left')
            self.audio_prob_bars[label] = {'canvas': canvas, 'val': val_lbl,
                                           'color': audio_colors[label]}

        # Bottom bar
        bottom = tk.Frame(main, bg=COLORS['bg_dark'])
        bottom.pack(fill='x', pady=(10, 0))

        self.cal_btn = tk.Button(
            bottom, text="Calibrate Face", command=self.start_calibration,
            font=('Helvetica', 10), bg=COLORS['accent_blue'],
            fg='white', padx=15, pady=5)
        self.cal_btn.pack(side='left', padx=5)

        self.load_btn = tk.Button(
            bottom, text="Load Profile", command=self.load_profile,
            font=('Helvetica', 10), bg=COLORS['bg_medium'],
            fg='white', padx=15, pady=5)
        self.load_btn.pack(side='left', padx=5)

        self.save_btn = tk.Button(
            bottom, text="Save Profile", command=self.save_profile,
            font=('Helvetica', 10), bg=COLORS['bg_medium'],
            fg='white', padx=15, pady=5, state='disabled')
        self.save_btn.pack(side='left', padx=5)

        self.metrics_label = tk.Label(
            bottom, text="Face: --ms | FPS: --",
            font=('Helvetica', 10), bg=COLORS['bg_dark'],
            fg=COLORS['text_gray'])
        self.metrics_label.pack(side='right')

    # ========================================================================
    # Calibration (face only)
    # ========================================================================

    def start_calibration(self):
        user_id = simpledialog.askstring(
            "User ID", "Enter user ID:", initialvalue="test_user")
        if not user_id:
            return
        self.current_user = user_id
        self.user_label.config(text=f"User: {user_id}")
        self.cal_in_progress = True
        self.cal_state_idx = 0
        self.captured_frames = []
        self.face_probs_ema = None  # Reset EMA on new calibration
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
        self.status_label.config(
            text=f"Calibrating: {state['label']} ({state['duration']}s)")

    def _capture_frame_for_calibration(self, result: Dict):
        if not self.cal_in_progress or self.cal_state_idx >= len(CALIBRATION_STATES):
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
            messagebox.showwarning("Retry",
                                   f"Not enough frames for {state['label']}.")
            self._start_state_capture()
            return

        frames = self.captured_frames[-FRAMES_TO_AVERAGE:]
        embeddings = [f['embedding'] for f in frames]
        avg_emb = average_embeddings(embeddings)

        if self.cal_state_idx == 0:
            self._temp_baseline = GenericBaseline(
                user_id=self.current_user, modality='deepface_emb')
            self._temp_baseline.REQUIRED_STATES = ['neutral', 'happy']

        self._temp_baseline.add_state(state['name'], avg_emb)
        self.cal_state_idx += 1
        self.captured_frames = []
        self.root.after(500, self._start_state_capture)

    def _complete_calibration(self):
        self.cal_in_progress = False
        self.progress_bar.pack_forget()
        self.instruction_label.config(text="")
        self.cal_detector.set_baseline(self._temp_baseline)

        # 2-state adaptive thresholds
        neutral_emb = self._temp_baseline.get_embedding('neutral')
        happy_emb = self._temp_baseline.get_embedding('happy')
        sim_nh = cosine_similarity(neutral_emb, happy_emb)
        thresholds = {
            'similarity_threshold': max(0.65, min(0.95, sim_nh * 0.85)),
            'neutral_threshold': max(0.65, min(0.95, sim_nh * 0.87)),
            'deviation_floor': max(0.50, min(0.90, sim_nh * 0.78)),
            'raw_override_confidence': 0.60,
        }
        self.cal_detector.set_adaptive_thresholds(thresholds)
        print(f"[Calibration] sim_neutral_happy={sim_nh:.3f}")
        print(f"  threshold={thresholds['similarity_threshold']:.3f} "
              f"neutral={thresholds['neutral_threshold']:.3f} "
              f"floor={thresholds['deviation_floor']:.3f}")

        self.cal_btn.config(state='normal')
        self.load_btn.config(state='normal')
        self.save_btn.config(state='normal')
        self.status_label.config(text="Calibration complete!")
        messagebox.showinfo("Done",
                            f"Face calibration complete for '{self.current_user}'.")

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
        user_id = simpledialog.askstring(
            "Load", f"Available: {', '.join(profiles)}\n\nEnter user ID:")
        if not user_id:
            return
        baseline = self.cal_manager.load_profile(user_id)
        if baseline is None:
            messagebox.showwarning("Not Found", f"Profile '{user_id}' not found.")
            return
        self.current_user = user_id
        self.user_label.config(text=f"User: {user_id}")
        self.cal_detector.set_baseline(baseline)
        self._temp_baseline = baseline

        neutral_emb = baseline.get_embedding('neutral')
        happy_emb = baseline.get_embedding('happy')
        if neutral_emb is not None and happy_emb is not None:
            sim_nh = cosine_similarity(neutral_emb, happy_emb)
            thresholds = {
                'similarity_threshold': max(0.65, min(0.95, sim_nh * 0.85)),
                'neutral_threshold': max(0.65, min(0.95, sim_nh * 0.87)),
                'deviation_floor': max(0.50, min(0.90, sim_nh * 0.78)),
                'raw_override_confidence': 0.60,
            }
            self.cal_detector.set_adaptive_thresholds(thresholds)

        self.save_btn.config(state='normal')
        self.status_label.config(text=f"Loaded profile for '{user_id}'")

    # ========================================================================
    # DeepFace face analysis
    # ========================================================================

    def _analyze_face(self, face_bgr):
        """Get probs from DeepFace full pipeline + embedding from sub-model."""
        analysis = self.face_extractor._deepface.analyze(
            face_bgr, actions=['emotion'],
            detector_backend='skip',
            enforce_detection=False,
            silent=True,
        )
        raw_probs = analysis[0]['emotion']
        emotion_probs = {
            'Anger': raw_probs.get('angry', 0.0) / 100,
            'Disgust': raw_probs.get('disgust', 0.0) / 100,
            'Fear': raw_probs.get('fear', 0.0) / 100,
            'Happiness': raw_probs.get('happy', 0.0) / 100,
            'Sadness': raw_probs.get('sad', 0.0) / 100,
            'Surprise': raw_probs.get('surprise', 0.0) / 100,
            'Neutral': raw_probs.get('neutral', 0.0) / 100,
        }

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        preprocessed = self.face_extractor._preprocess(face_rgb)
        embedding = self.face_extractor._embedding_model.predict(
            preprocessed, verbose=0)[0]

        top_emotion = max(emotion_probs, key=emotion_probs.get)
        return {
            'embedding': embedding,
            'emotion_probs': emotion_probs,
            'top_emotion': top_emotion,
            'confidence': emotion_probs[top_emotion],
        }

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

    def update_fusion_display(self):
        """Read latest results and update all display elements."""
        with self._lock:
            face_for_fusion = self._latest_face_for_fusion.copy() if self._latest_face_for_fusion else None
            face_raw = self._latest_face_raw.copy() if self._latest_face_raw else None
            face_cal = self._latest_face_cal.copy() if self._latest_face_cal else None
            audio_result = self._latest_audio_result.copy() if self._latest_audio_result else None
            face_ts = self._face_timestamp
            audio_ts = self._audio_timestamp

        now = time.time()
        face_stale = (now - face_ts) > 1.0 if face_ts > 0 else True
        audio_stale = (now - audio_ts) > AUDIO_STALE_THRESHOLD if audio_ts > 0 else True

        # Face column
        if face_cal and not face_stale:
            source = face_cal.get('emotion_source', '')
            tag = {'calibration': '[CAL]', 'raw_model': '[RAW]',
                   'fallback': '[FB]', 'deviation_fallback': '[DEV]'
                   }.get(source, f'[{source}]')
            self.face_emotion_lbl.config(text=face_cal.get('emotion', '--'))
            self.face_info_lbl.config(
                text=f"{face_cal.get('confidence', 0):.0%} {tag}")
        else:
            self.face_emotion_lbl.config(text="--")
            self.face_info_lbl.config(text="No face" if face_stale else "--")

        # Audio column
        if audio_result and not audio_stale:
            self.audio_emotion_lbl.config(text=audio_result.get('top_emotion', '--'))
            self.audio_info_lbl.config(
                text=f"{audio_result.get('confidence', 0):.0%}")
        else:
            self.audio_emotion_lbl.config(text="--")
            self.audio_info_lbl.config(text="No audio" if audio_stale else "--")

        # Fusion
        face_input = face_for_fusion if (face_for_fusion and not face_stale) else None
        audio_input = audio_result if (audio_result and not audio_stale) else None

        fusion_result = self.fusion.fuse(face_input, audio_input)

        self.fused_emotion_lbl.config(text=fusion_result.emotion)
        self.fused_info_lbl.config(
            text=f"{fusion_result.confidence:.0%} | {fusion_result.quadrant_label}")
        self.weights_lbl.config(
            text=f"Face: {fusion_result.face_weight:.0%} | "
                 f"Audio: {fusion_result.audio_weight:.0%} | "
                 f"{fusion_result.modalities_present}")

        # Adapter source
        adapter_src = face_for_fusion.get('_face_source', '--') if face_for_fusion else '--'
        self.adapter_lbl.config(text=f"Adapter: {adapter_src}")

        # Similarity bars
        if face_cal and face_cal.get('calibrated'):
            sims = face_cal.get('similarities', {})
            closest = face_cal.get('closest_baseline', '')
            for state in ['neutral', 'happy']:
                if state in sims:
                    sim = sims[state]
                    bars = self.sim_bars[state]
                    bars['canvas'].delete('all')
                    bw = max(0, min(160, int(160 * sim)))
                    if bw > 0:
                        fill = bars['color'] if state == closest else '#475569'
                        bars['canvas'].create_rectangle(
                            0, 0, bw, 16, fill=fill, outline='')
                    bars['label'].config(text=f"{sim:.3f}")
                    bars['label'].config(
                        fg=bars['color'] if state == closest else COLORS['text_gray'])
        else:
            for state in ['neutral', 'happy']:
                self.sim_bars[state]['canvas'].delete('all')
                self.sim_bars[state]['label'].config(text="--")

        # Raw face probability bars (before adapter)
        if face_raw:
            rprobs = face_raw.get('emotion_probs', {})
            for label, bars in self.raw_prob_bars.items():
                p = rprobs.get(label, 0.0)
                bars['canvas'].delete('all')
                bw = max(0, min(100, int(100 * p)))
                if bw > 0:
                    bars['canvas'].create_rectangle(
                        0, 0, bw, 12, fill=bars['color'], outline='')
                bars['val'].config(text=f"{p:.0%}")
        else:
            for bars in self.raw_prob_bars.values():
                bars['canvas'].delete('all')
                bars['val'].config(text="--")

        # Adapted face probability bars (what fusion sees)
        if face_for_fusion:
            probs = face_for_fusion.get('emotion_probs', {})
            for label, bars in self.prob_bars.items():
                p = probs.get(label, 0.0)
                bars['canvas'].delete('all')
                bw = max(0, min(140, int(140 * p)))
                if bw > 0:
                    bars['canvas'].create_rectangle(
                        0, 0, bw, 14, fill=bars['color'], outline='')
                bars['val'].config(text=f"{p:.0%}")
        else:
            for bars in self.prob_bars.values():
                bars['canvas'].delete('all')
                bars['val'].config(text="--")

        # Audio probability bars
        if audio_result and not audio_stale:
            aprobs = audio_result.get('emotion_probs', {})
            for label, bars in self.audio_prob_bars.items():
                p = aprobs.get(label, 0.0)
                bars['canvas'].delete('all')
                bw = max(0, min(140, int(140 * p)))
                if bw > 0:
                    bars['canvas'].create_rectangle(
                        0, 0, bw, 14, fill=bars['color'], outline='')
                bars['val'].config(text=f"{p:.0%}")
        else:
            for bars in self.audio_prob_bars.values():
                bars['canvas'].delete('all')
                bars['val'].config(text="--")

        # Schedule next update
        if self.running:
            self.root.after(100, self.update_fusion_display)

    # ========================================================================
    # Processing Loops
    # ========================================================================

    def face_loop(self):
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(0.1)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                face_bgr, bbox = self.face_detector.detect(frame)

                if face_bgr is not None:
                    self.no_face_count = 0
                    start = time.time()
                    result = self._analyze_face(face_bgr)
                    self.face_latency = time.time() - start

                    if self.cal_in_progress:
                        self.root.after(
                            0, lambda r=result:
                            self._capture_frame_for_calibration(r))

                    # Get raw and calibrated predictions
                    raw_pred = self.cal_detector.get_raw_prediction(result)
                    cal_pred = self.cal_detector.get_calibrated_prediction(result)

                    # Build adapted face result for fusion
                    face_for_fusion = build_face_result(raw_pred, cal_pred)

                    # EMA-smooth the probability vector to dampen speech jitter
                    probs_array = np.array([
                        face_for_fusion['emotion_probs'].get(em, 0.0)
                        for em in EMOTION_LABELS])
                    if self.face_probs_ema is None:
                        self.face_probs_ema = probs_array.copy()
                    else:
                        self.face_probs_ema = (
                            self.face_ema_decay * probs_array
                            + (1 - self.face_ema_decay) * self.face_probs_ema)
                    # Rebuild face_for_fusion with smoothed probs
                    smoothed_probs = {
                        em: float(self.face_probs_ema[i])
                        for i, em in enumerate(EMOTION_LABELS)}
                    smoothed_top = max(smoothed_probs, key=smoothed_probs.get)
                    face_for_fusion = {
                        'top_emotion': smoothed_top,
                        'confidence': smoothed_probs[smoothed_top],
                        'emotion_probs': smoothed_probs,
                        '_face_source': face_for_fusion.get('_face_source', ''),
                    }

                    with self._lock:
                        self._latest_face_for_fusion = face_for_fusion
                        self._latest_face_raw = raw_pred
                        self._latest_face_cal = cal_pred
                        self._face_timestamp = time.time()

                    self.root.after(
                        0, lambda f=frame.copy(), b=bbox:
                        self.update_video(f, b))
                else:
                    self.no_face_count += 1
                    self.root.after(
                        0, lambda f=frame.copy():
                        self.update_video(f, None))
                    if self.no_face_count >= self.no_face_grace:
                        self.face_probs_ema = None  # Reset EMA
                        with self._lock:
                            self._latest_face_for_fusion = None
                            self._latest_face_raw = None
                            self._latest_face_cal = None

                # FPS
                self._face_frame_count += 1
                elapsed = time.time() - self._face_fps_time
                if elapsed >= 1.0:
                    self.face_fps = self._face_frame_count / elapsed
                    self._face_frame_count = 0
                    self._face_fps_time = time.time()
                    self.root.after(
                        0, lambda: self.metrics_label.config(
                            text=f"Face: {self.face_latency*1000:.0f}ms | "
                                 f"FPS: {self.face_fps:.1f}"))

                time.sleep(0.01)

            except Exception as e:
                print(f"Face loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def audio_loop(self):
        print("[Audio] Loop started, waiting for chunks...")
        first_result = True
        while self.running:
            try:
                chunk = self.audio_capture.get_chunk(timeout=0.5)
                if chunk is None:
                    continue

                result = self.audio_extractor.extract(chunk, SAMPLE_RATE)

                if first_result:
                    print(f"[Audio] First result: {result.get('top_emotion')} "
                          f"{result.get('confidence', 0):.0%}")
                    first_result = False

                with self._lock:
                    self._latest_audio_result = result
                    self._audio_timestamp = time.time()

            except Exception as e:
                print(f"Audio loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)

    # ========================================================================
    # Lifecycle
    # ========================================================================

    def on_close(self):
        self.running = False
        self.audio_capture.stop()
        if self.cap:
            self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        def init():
            self.root.after(0, lambda: self.status_label.config(
                text="Loading DeepFace model..."))
            self.face_extractor.load(
                status_callback=lambda msg: self.root.after(
                    0, lambda m=msg: self.status_label.config(text=m)))

            self.root.after(0, lambda: self.status_label.config(
                text="Loading Emotion2Vec model..."))
            self.audio_extractor.load(
                status_callback=lambda msg: self.root.after(
                    0, lambda m=msg: self.status_label.config(text=m)))

            self.root.after(0, lambda: self.status_label.config(
                text=f"Starting camera {self.camera_index}..."))
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            time.sleep(0.5)

            self.root.after(0, lambda: self.status_label.config(
                text="Starting audio capture..."))
            try:
                self.audio_capture.start()
            except Exception as e:
                print(f"Audio start failed: {e}")
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Audio failed: {e}. Face-only mode."))

            self.running = True
            self.root.after(0, lambda: self.status_label.config(
                text="Ready! Click 'Calibrate Face' to start."))

            # Start processing threads
            threading.Thread(target=self.face_loop, daemon=True).start()
            threading.Thread(target=self.audio_loop, daemon=True).start()

            # Start fusion display update loop
            self.root.after(500, self.update_fusion_display)

        threading.Thread(target=init, daemon=True).start()
        self.root.mainloop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='DeepFace + Emotion2Vec Fusion Demo')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (0=default, 1=MacBook webcam)')
    args = parser.parse_args()

    print(f"Using camera index: {args.camera}")
    app = DeepFaceAudioFusionApp(camera_index=args.camera)
    app.run()
