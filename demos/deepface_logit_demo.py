"""
DeepFace Neutral/Smile Calibration Demo

Calibrates the neutral-to-positive boundary using a scalar smile score:
    score = log(p_happy) - log(p_neutral)

This collapses the 7-class problem into a 1-D decision where the signal
actually lives. Positive score = more happy, negative = more neutral.

Calibration captures two states:
  1. Neutral face -> establishes the user's resting-face score
  2. Subtle smile -> establishes the user's mild-positive score

If the two scores separate sufficiently, thresholds are set from the data.
At inference:
  - score >= smile_threshold  -> Happiness (calibration override)
  - score <= neutral_threshold -> Neutral  (calibration override)
  - otherwise                  -> raw DeepFace (ambiguous zone)

Non-neutral/non-happy emotions (anger, sadness, etc.) always use raw DeepFace.

Usage:
    python demos/deepface_logit_demo.py --camera 1
"""

import time
import threading
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
from typing import Optional, List, Dict


# ============================================================================
# Configuration
# ============================================================================

EMOTION_LABELS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

EMOTION_COLORS = {
    'Anger': '#E74C3C',
    'Disgust': '#8E44AD',
    'Fear': '#9B59B6',
    'Happiness': '#F1C40F',
    'Sadness': '#3498DB',
    'Surprise': '#E67E22',
    'Neutral': '#95A5A6',
}

HAPPY_IDX = EMOTION_LABELS.index('Happiness')
NEUTRAL_IDX = EMOTION_LABELS.index('Neutral')

FRAMES_TO_AVERAGE = 25

CALIBRATION_STATES = [
    {'name': 'neutral', 'label': 'Neutral', 'duration': 5,
     'instruction': 'Look at the camera with a relaxed, natural expression.\n'
                    'This captures your resting-face baseline.'},
    {'name': 'subtle_smile', 'label': 'Subtle Smile', 'duration': 5,
     'instruction': 'Show a gentle, natural smile — not a big grin.\n'
                    'Think of something mildly pleasant.'},
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
# Smile Score
# ============================================================================

def smile_score_from_probs(probs: np.ndarray, eps: float = 1e-4) -> float:
    """Compute smile score as log-odds ratio: log(p_happy / p_neutral).

    Positive = face looks more happy than neutral.
    Negative = face looks more neutral than happy.
    Zero = equal probability.
    """
    return float(np.log(probs[HAPPY_IDX] + eps) - np.log(probs[NEUTRAL_IDX] + eps))


# ============================================================================
# Calibrator
# ============================================================================

class NeutralSmileCalibrator:
    """Calibrates the neutral/smile boundary from captured probability distributions.

    Computes smile_score for each calibration frame, then sets thresholds
    based on the separation between neutral and smile score distributions.
    """

    def __init__(self):
        self.valid = False
        self.neutral_mean: Optional[float] = None
        self.neutral_std: Optional[float] = None
        self.smile_mean: Optional[float] = None
        self.smile_std: Optional[float] = None
        self.neutral_threshold: Optional[float] = None
        self.smile_threshold: Optional[float] = None
        self.gap: Optional[float] = None

    def calibrate(self, neutral_probs_list: List[np.ndarray],
                  smile_probs_list: List[np.ndarray]) -> Dict:
        """Compute thresholds from captured calibration data.

        Returns diagnostic dict with gap, spread, validity, etc.
        """
        neutral_scores = np.array(
            [smile_score_from_probs(p) for p in neutral_probs_list])
        smile_scores = np.array(
            [smile_score_from_probs(p) for p in smile_probs_list])

        self.neutral_mean = float(neutral_scores.mean())
        self.neutral_std = float(neutral_scores.std())
        self.smile_mean = float(smile_scores.mean())
        self.smile_std = float(smile_scores.std())

        self.gap = self.smile_mean - self.neutral_mean
        spread = max(0.05, self.neutral_std, self.smile_std)
        midpoint = (self.neutral_mean + self.smile_mean) / 2

        self.neutral_threshold = midpoint - spread
        self.smile_threshold = midpoint + spread

        # Valid only if the two states actually separate
        self.valid = self.gap > max(0.20, 2 * spread)

        return {
            'gap': self.gap,
            'spread': spread,
            'midpoint': midpoint,
            'valid': self.valid,
            'neutral_mean': self.neutral_mean,
            'neutral_std': self.neutral_std,
            'smile_mean': self.smile_mean,
            'smile_std': self.smile_std,
            'neutral_threshold': self.neutral_threshold,
            'smile_threshold': self.smile_threshold,
        }

    def classify(self, score: float, raw_top: str) -> tuple:
        """Classify based on smile score.

        Returns (label, source) where source is one of:
            'smile_cal', 'neutral_cal', 'raw_ambiguous', 'raw_non_target', 'raw'

        Calibration only overrides when raw DeepFace already says Neutral or
        Happiness. All other emotions (anger, sadness, surprise, etc.) pass
        through untouched — calibration never erases non-target emotions.
        """
        if not self.valid:
            return raw_top, 'raw'

        # Only override at the neutral/happiness boundary
        if raw_top not in ('Neutral', 'Happiness'):
            return raw_top, 'raw_non_target'

        if score >= self.smile_threshold:
            return 'Happiness', 'smile_cal'
        elif score <= self.neutral_threshold:
            return 'Neutral', 'neutral_cal'
        else:
            return raw_top, 'raw_ambiguous'


# ============================================================================
# Face Detector
# ============================================================================

class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame):
        """Detect largest face and return BGR crop (matching DeepFace input)."""
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
# DeepFace Extractor
# ============================================================================

class DeepFaceExtractor:
    """Extracts emotion probabilities from DeepFace."""

    def __init__(self, detector_backend='skip'):
        self.detector_backend = detector_backend
        self._deepface = None

    def load(self, status_callback=None):
        if status_callback:
            status_callback("Loading DeepFace...")
        from deepface import DeepFace
        self._deepface = DeepFace
        # Warm up
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        try:
            self._deepface.analyze(
                dummy, actions=['emotion'],
                detector_backend='skip', enforce_detection=False)
        except Exception:
            pass
        if status_callback:
            status_callback("DeepFace loaded!")

    def extract(self, face_image) -> Dict:
        """Extract emotion probabilities from a BGR face image."""
        if self._deepface is None:
            self.load()
        analysis = self._deepface.analyze(
            face_image, actions=['emotion'],
            detector_backend=self.detector_backend,
            enforce_detection=False
        )
        raw = analysis[0]['emotion']
        # Match EMOTION_LABELS order, convert percentages to probabilities
        probs = np.array([
            raw.get('angry', 0.0) / 100,
            raw.get('disgust', 0.0) / 100,
            raw.get('fear', 0.0) / 100,
            raw.get('happy', 0.0) / 100,
            raw.get('sad', 0.0) / 100,
            raw.get('surprise', 0.0) / 100,
            raw.get('neutral', 0.0) / 100,
        ])
        top_idx = int(np.argmax(probs))
        return {
            'probs': probs,
            'top_emotion': EMOTION_LABELS[top_idx],
            'confidence': float(probs[top_idx]),
        }


# ============================================================================
# Main Application
# ============================================================================

class DeepFaceSmileDemoApp:
    """GUI for testing neutral/smile calibration."""

    def __init__(self, camera_index: int = 0):
        self.extractor = DeepFaceExtractor()
        self.face_detector = FaceDetector()
        self.calibrator = NeutralSmileCalibrator()

        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.current_user: Optional[str] = None

        # Calibration state
        self.cal_in_progress = False
        self.cal_state_idx = 0
        self.cal_start_time = 0.0
        self.captured_probs: Dict[str, List[np.ndarray]] = {
            'neutral': [], 'subtle_smile': [],
        }

        # EMA smoothing on raw probs (smooth the signal, not just labels)
        self.probs_ema: Optional[np.ndarray] = None
        self.ema_decay = 0.3  # weight of new frame (lower = smoother)

        # Metrics
        self.inference_time = 0.0
        self.fps = 0.0
        self.frame_count = 0
        self.fps_start_time = time.time()

        # GUI
        self.root = tk.Tk()
        self.root.title("DeepFace Neutral/Smile Calibration")
        self.root.geometry("1200x750")
        self.root.configure(bg=COLORS['bg_dark'])
        self._setup_ui()

    def _setup_ui(self):
        main = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main.pack(fill='both', expand=True, padx=10, pady=10)

        # Top bar
        top = tk.Frame(main, bg=COLORS['bg_dark'])
        top.pack(fill='x', pady=(0, 10))
        tk.Label(top, text="DeepFace Neutral/Smile Calibration",
                 font=('Helvetica', 18, 'bold'),
                 bg=COLORS['bg_dark'], fg=COLORS['text_white']).pack(side='left')
        self.user_label = tk.Label(top, text="[No User]",
                                   font=('Helvetica', 12),
                                   bg=COLORS['bg_dark'], fg=COLORS['text_gray'])
        self.user_label.pack(side='right')

        # Content
        content = tk.Frame(main, bg=COLORS['bg_dark'])
        content.pack(fill='both', expand=True, pady=10)

        # Left — camera
        left = tk.Frame(content, bg=COLORS['bg_dark'])
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

        # Right — analysis panel
        right = tk.Frame(content, bg=COLORS['bg_medium'], padx=15, pady=10)
        right.pack(side='right', fill='both', expand=True, padx=(10, 0))

        # Emotion comparison row
        comp = tk.Frame(right, bg=COLORS['bg_medium'])
        comp.pack(fill='x', pady=(0, 8))

        raw_box = tk.Frame(comp, bg=COLORS['bg_dark'], padx=10, pady=8)
        raw_box.pack(side='left', fill='both', expand=True, padx=(0, 5))
        tk.Label(raw_box, text="RAW DEEPFACE",
                 font=('Helvetica', 11, 'bold'),
                 bg=COLORS['bg_dark'], fg=COLORS['accent_red']).pack()
        self.raw_emotion = tk.Label(raw_box, text="--",
                                    font=('Helvetica', 20, 'bold'),
                                    bg=COLORS['bg_dark'],
                                    fg=COLORS['text_white'])
        self.raw_emotion.pack(pady=5)
        self.raw_confidence = tk.Label(raw_box, text="--",
                                       font=('Helvetica', 10),
                                       bg=COLORS['bg_dark'],
                                       fg=COLORS['text_gray'])
        self.raw_confidence.pack()

        cal_box = tk.Frame(comp, bg=COLORS['bg_dark'], padx=10, pady=8)
        cal_box.pack(side='right', fill='both', expand=True, padx=(5, 0))
        tk.Label(cal_box, text="CALIBRATED",
                 font=('Helvetica', 11, 'bold'),
                 bg=COLORS['bg_dark'], fg=COLORS['accent_green']).pack()
        self.cal_emotion = tk.Label(cal_box, text="--",
                                    font=('Helvetica', 20, 'bold'),
                                    bg=COLORS['bg_dark'],
                                    fg=COLORS['text_white'])
        self.cal_emotion.pack(pady=5)
        self.cal_source = tk.Label(cal_box, text="--",
                                   font=('Helvetica', 10),
                                   bg=COLORS['bg_dark'],
                                   fg=COLORS['text_gray'])
        self.cal_source.pack()

        # Smile score gauge
        gauge_frame = tk.Frame(right, bg=COLORS['bg_medium'])
        gauge_frame.pack(fill='x', pady=(5, 8))

        tk.Label(gauge_frame, text="SMILE SCORE (log-odds)",
                 font=('Helvetica', 10, 'bold'),
                 bg=COLORS['bg_medium'], fg=COLORS['text_white']).pack(pady=(0, 3))

        gauge_row = tk.Frame(gauge_frame, bg=COLORS['bg_medium'])
        gauge_row.pack(fill='x')
        tk.Label(gauge_row, text="Neutral", font=('Helvetica', 9),
                 bg=COLORS['bg_medium'], fg='#95A5A6').pack(side='left')
        self.gauge_canvas = tk.Canvas(gauge_row, width=300, height=30,
                                      bg='#1e293b', highlightthickness=0)
        self.gauge_canvas.pack(side='left', padx=8)
        tk.Label(gauge_row, text="Smile", font=('Helvetica', 9),
                 bg=COLORS['bg_medium'], fg='#F1C40F').pack(side='left')

        self.score_label = tk.Label(gauge_frame, text="Score: --",
                                    font=('Courier', 10),
                                    bg=COLORS['bg_medium'],
                                    fg=COLORS['text_white'])
        self.score_label.pack()

        # Calibration diagnostics
        diag_frame = tk.Frame(right, bg=COLORS['bg_medium'])
        diag_frame.pack(fill='x', pady=(5, 8))
        tk.Label(diag_frame, text="CALIBRATION",
                 font=('Helvetica', 10, 'bold'),
                 bg=COLORS['bg_medium'], fg=COLORS['text_white']).pack(pady=(0, 3))
        self.diag_label = tk.Label(diag_frame, text="[Not calibrated]",
                                   font=('Courier', 9),
                                   bg=COLORS['bg_medium'],
                                   fg=COLORS['text_gray'],
                                   justify='left', anchor='w')
        self.diag_label.pack(fill='x')

        # Raw probability bars (single column — calibration is an overlay, not a new distribution)
        dist = tk.Frame(right, bg=COLORS['bg_medium'])
        dist.pack(fill='both', expand=True, pady=(5, 0))
        tk.Label(dist, text="RAW PROBABILITIES",
                 font=('Helvetica', 10, 'bold'),
                 bg=COLORS['bg_medium'], fg=COLORS['text_white']).pack(pady=(0, 4))

        self.prob_bars = {}
        for label in EMOTION_LABELS:
            row = tk.Frame(dist, bg=COLORS['bg_medium'])
            row.pack(fill='x', pady=1)
            tk.Label(row, text=label[:8], font=('Helvetica', 9),
                     width=10, anchor='e',
                     bg=COLORS['bg_medium'],
                     fg=COLORS['text_gray']).pack(side='left')
            canvas = tk.Canvas(row, width=160, height=14,
                               bg='#1e293b', highlightthickness=0)
            canvas.pack(side='left', padx=(8, 2))
            val_lbl = tk.Label(row, text="--", font=('Helvetica', 8),
                               width=5, bg=COLORS['bg_medium'],
                               fg=COLORS['text_gray'])
            val_lbl.pack(side='left')
            self.prob_bars[label] = {'canvas': canvas, 'val': val_lbl}

        # Bottom bar
        bottom = tk.Frame(main, bg=COLORS['bg_dark'])
        bottom.pack(fill='x', pady=(10, 0))

        self.cal_btn = tk.Button(
            bottom, text="Calibrate", command=self.start_calibration,
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
            bottom, text="Latency: -- ms | FPS: --",
            font=('Helvetica', 10), bg=COLORS['bg_dark'],
            fg=COLORS['text_gray'])
        self.metrics_label.pack(side='right')

    # ========================================================================
    # Calibration
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
        self.captured_probs = {'neutral': [], 'subtle_smile': []}
        self._reset_ema()
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
        self.cal_start_time = time.time()
        self.status_label.config(
            text=f"Calibrating: {state['label']} ({state['duration']}s)")

    def _capture_frame_for_calibration(self, result: Dict):
        if not self.cal_in_progress:
            return
        if self.cal_state_idx >= len(CALIBRATION_STATES):
            return
        state = CALIBRATION_STATES[self.cal_state_idx]
        elapsed = time.time() - self.cal_start_time
        self.progress_var.set(min(100, (elapsed / state['duration']) * 100))
        if elapsed > 1.0:  # skip first second (settling)
            self.captured_probs[state['name']].append(result['probs'])
        if elapsed >= state['duration']:
            self._finalize_state_capture()

    def _finalize_state_capture(self):
        state = CALIBRATION_STATES[self.cal_state_idx]
        frames = self.captured_probs[state['name']]
        if len(frames) < 5:
            messagebox.showwarning("Retry",
                                   f"Not enough frames for {state['label']}.")
            self.cal_start_time = time.time()
            self.captured_probs[state['name']] = []
            return

        # Keep last N frames
        self.captured_probs[state['name']] = frames[-FRAMES_TO_AVERAGE:]
        self.cal_state_idx += 1
        self.root.after(500, self._start_state_capture)

    def _complete_calibration(self):
        self.cal_in_progress = False
        self.progress_bar.pack_forget()
        self.instruction_label.config(text="")

        diag = self.calibrator.calibrate(
            self.captured_probs['neutral'],
            self.captured_probs['subtle_smile'],
        )

        # Console diagnostics
        print(f"\n[Neutral/Smile Calibration]")
        print(f"  Neutral score: mean={diag['neutral_mean']:.3f} "
              f"std={diag['neutral_std']:.3f}")
        print(f"  Smile score:   mean={diag['smile_mean']:.3f} "
              f"std={diag['smile_std']:.3f}")
        print(f"  Gap: {diag['gap']:.3f}  Spread: {diag['spread']:.3f}")
        print(f"  Thresholds: neutral<={diag['neutral_threshold']:.3f}  "
              f"smile>={diag['smile_threshold']:.3f}")
        print(f"  Valid: {diag['valid']}")

        # Update GUI diagnostics
        self._update_diag_display(diag)

        self.cal_btn.config(state='normal')
        self.load_btn.config(state='normal')
        self.save_btn.config(state='normal')

        if diag['valid']:
            self.status_label.config(
                text="Calibration valid! Showing calibrated output.")
            messagebox.showinfo(
                "Done",
                f"Calibration complete for '{self.current_user}'.\n"
                f"Gap: {diag['gap']:.3f} — states are separable.")
        else:
            self.status_label.config(
                text="Calibration invalid — states too similar.")
            messagebox.showwarning(
                "Weak Calibration",
                f"Neutral and smile scores are too close "
                f"(gap={diag['gap']:.3f}).\n"
                f"Calibration will fall back to raw DeepFace.\n"
                f"Try a more distinct smile.")

    def _update_diag_display(self, diag: Dict):
        valid_str = "YES" if diag['valid'] else "NO (insufficient separation)"
        valid_color = COLORS['accent_green'] if diag['valid'] else COLORS['accent_red']
        self.diag_label.config(
            text=f"  Neutral mean:  {diag['neutral_mean']:+.3f}\n"
                 f"  Smile mean:    {diag['smile_mean']:+.3f}\n"
                 f"  Gap:           {diag['gap']:.3f}\n"
                 f"  Neutral thr:  <={diag['neutral_threshold']:+.3f}\n"
                 f"  Smile thr:    >={diag['smile_threshold']:+.3f}\n"
                 f"  Valid:          {valid_str}",
            fg=valid_color)

    # ========================================================================
    # Profile persistence
    # ========================================================================

    def save_profile(self):
        if self.calibrator.neutral_mean is None:
            messagebox.showwarning("No Calibration", "Nothing to save.")
            return
        import pickle, os
        os.makedirs('./user_profiles', exist_ok=True)
        filepath = f"./user_profiles/{self.current_user}_smile.pkl"
        data = {
            'user_id': self.current_user,
            'neutral_mean': self.calibrator.neutral_mean,
            'neutral_std': self.calibrator.neutral_std,
            'smile_mean': self.calibrator.smile_mean,
            'smile_std': self.calibrator.smile_std,
            'neutral_threshold': self.calibrator.neutral_threshold,
            'smile_threshold': self.calibrator.smile_threshold,
            'gap': self.calibrator.gap,
            'valid': self.calibrator.valid,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        messagebox.showinfo("Saved", f"Profile saved to:\n{filepath}")

    def load_profile(self):
        import os
        profiles_dir = './user_profiles'
        if not os.path.exists(profiles_dir):
            messagebox.showinfo("No Profiles", "No saved profiles found.")
            return
        profiles = [f.replace('_smile.pkl', '')
                    for f in os.listdir(profiles_dir)
                    if f.endswith('_smile.pkl')]
        if not profiles:
            messagebox.showinfo("No Profiles", "No smile profiles found.")
            return
        user_id = simpledialog.askstring(
            "Load", f"Available: {', '.join(profiles)}\n\nEnter user ID:")
        if not user_id:
            return
        filepath = f"{profiles_dir}/{user_id}_smile.pkl"
        if not os.path.exists(filepath):
            messagebox.showwarning("Not Found",
                                   f"Profile '{user_id}' not found.")
            return
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.current_user = user_id
        self.user_label.config(text=f"User: {user_id}")
        self._reset_ema()
        self.calibrator.neutral_mean = data['neutral_mean']
        self.calibrator.neutral_std = data['neutral_std']
        self.calibrator.smile_mean = data['smile_mean']
        self.calibrator.smile_std = data['smile_std']
        self.calibrator.neutral_threshold = data['neutral_threshold']
        self.calibrator.smile_threshold = data['smile_threshold']
        self.calibrator.gap = data['gap']
        self.calibrator.valid = data['valid']
        self.save_btn.config(state='normal')
        self._update_diag_display(data)
        self.status_label.config(
            text=f"Loaded profile for '{user_id}' "
                 f"({'valid' if data['valid'] else 'invalid'})")

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

    def _ema_smooth(self, new_probs: np.ndarray) -> np.ndarray:
        """EMA on probability vectors — smooth the signal, not the labels."""
        if self.probs_ema is None:
            self.probs_ema = new_probs.copy()
        else:
            self.probs_ema = (self.ema_decay * new_probs
                              + (1 - self.ema_decay) * self.probs_ema)
        return self.probs_ema

    def _draw_gauge(self, score: float):
        """Draw the smile score gauge with threshold markers and zones."""
        self.gauge_canvas.delete('all')
        w, h = 300, 30

        # Determine display range from calibration data or defaults
        if self.calibrator.neutral_threshold is not None:
            lo = min(self.calibrator.neutral_threshold - 1.0, score - 0.5)
            hi = max(self.calibrator.smile_threshold + 1.0, score + 0.5)
        else:
            lo, hi = -6.0, 2.0

        def x_pos(val):
            return max(0, min(w, int(w * (val - lo) / (hi - lo))))

        # Draw colored zones if calibrated
        if self.calibrator.neutral_threshold is not None:
            nt = x_pos(self.calibrator.neutral_threshold)
            st = x_pos(self.calibrator.smile_threshold)
            # Neutral zone (left, blue-ish)
            self.gauge_canvas.create_rectangle(
                0, 0, nt, h, fill='#2c3e6b', outline='')
            # Ambiguous zone (middle, gray)
            self.gauge_canvas.create_rectangle(
                nt, 0, st, h, fill='#3d3d3d', outline='')
            # Smile zone (right, warm)
            self.gauge_canvas.create_rectangle(
                st, 0, w, h, fill='#4a3e1b', outline='')
            # Threshold lines
            self.gauge_canvas.create_line(
                nt, 0, nt, h, fill='#5b7db1', width=2)
            self.gauge_canvas.create_line(
                st, 0, st, h, fill='#b1a05b', width=2)

        # Score marker (white triangle + line)
        sx = x_pos(score)
        self.gauge_canvas.create_polygon(
            sx - 5, 0, sx + 5, 0, sx, 8,
            fill='#FFFFFF', outline='')
        self.gauge_canvas.create_line(
            sx, 8, sx, h, fill='#FFFFFF', width=2)

    def update_display(self, result: Dict):
        raw_probs = result['probs']

        # Single EMA-smoothed distribution drives everything
        smoothed = self._ema_smooth(raw_probs)
        raw_top_idx = int(np.argmax(smoothed))
        raw_top = EMOTION_LABELS[raw_top_idx]
        raw_conf = float(smoothed[raw_top_idx])

        # Smile score from smoothed probs
        score = smile_score_from_probs(smoothed)

        # Calibrated decision (overlay on raw)
        cal_label, source = self.calibrator.classify(score, raw_top)

        # Update emotion labels
        self.raw_emotion.config(text=raw_top)
        self.raw_confidence.config(text=f"{raw_conf:.0%}")

        source_tags = {
            'smile_cal': '[SMILE CAL]',
            'neutral_cal': '[NEUTRAL CAL]',
            'raw_ambiguous': '[AMBIGUOUS]',
            'raw_non_target': '[RAW PASS-THRU]',
            'raw': '[RAW]',
        }
        source_colors = {
            'smile_cal': COLORS['accent_yellow'],
            'neutral_cal': COLORS['accent_blue'],
            'raw_ambiguous': COLORS['text_gray'],
            'raw_non_target': COLORS['accent_red'],
            'raw': COLORS['text_gray'],
        }
        self.cal_emotion.config(
            text=cal_label,
            fg=source_colors.get(source, COLORS['text_white']))
        self.cal_source.config(text=source_tags.get(source, source))

        # Score display and gauge
        self.score_label.config(text=f"Score: {score:+.3f}")
        self._draw_gauge(score)

        # Raw probability bars (smoothed)
        for i, label in enumerate(EMOTION_LABELS):
            bars = self.prob_bars[label]
            p = float(smoothed[i])
            bars['canvas'].delete('all')
            bw = max(0, min(160, int(160 * p)))
            if bw > 0:
                bars['canvas'].create_rectangle(
                    0, 0, bw, 14,
                    fill=EMOTION_COLORS[label], outline='')
            bars['val'].config(text=f"{p:.0%}")

    def _reset_ema(self):
        """Reset EMA state to prevent stale history from leaking."""
        self.probs_ema = None

    def _clear_display(self):
        """Clear all live display elements."""
        self.gauge_canvas.delete('all')
        for label in EMOTION_LABELS:
            bars = self.prob_bars[label]
            bars['canvas'].delete('all')
            bars['val'].config(text="--")

    def show_no_face(self):
        self._reset_ema()
        self.raw_emotion.config(text="NO FACE")
        self.raw_confidence.config(text="--")
        self.cal_emotion.config(text="NO FACE", fg=COLORS['text_white'])
        self.cal_source.config(text="--")
        self.score_label.config(text="Score: --")
        self._clear_display()

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
                        self.root.after(
                            0, lambda r=result:
                            self._capture_frame_for_calibration(r))

                    self.root.after(
                        0, lambda f=frame.copy(), b=bbox:
                        self.update_video(f, b))
                    self.root.after(
                        0, lambda r=result: self.update_display(r))
                else:
                    self.root.after(
                        0, lambda f=frame.copy():
                        self.update_video(f, None))
                    self.root.after(0, self.show_no_face)

                # FPS tracking
                self.frame_count += 1
                elapsed = time.time() - self.fps_start_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.fps_start_time = time.time()
                    self.root.after(
                        0, lambda: self.metrics_label.config(
                            text=f"Latency: {self.inference_time*1000:.0f}ms"
                                 f" | FPS: {self.fps:.1f}"))

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
            self.root.after(0, lambda: self.status_label.config(
                text="Loading DeepFace model..."))
            self.extractor.load(
                status_callback=lambda msg: self.root.after(
                    0, lambda m=msg: self.status_label.config(text=m)))

            self.root.after(0, lambda: self.status_label.config(
                text=f"Starting camera {self.camera_index}..."))
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            time.sleep(0.5)

            self.running = True
            self.root.after(0, lambda: self.status_label.config(
                text="Ready! Click 'Calibrate' to start."))
            threading.Thread(target=self.main_loop, daemon=True).start()

        threading.Thread(target=init, daemon=True).start()
        self.root.mainloop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='DeepFace Neutral/Smile Calibration Demo')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (0=default, 1=MacBook webcam)')
    args = parser.parse_args()

    print(f"Using camera index: {args.camera}")
    app = DeepFaceSmileDemoApp(camera_index=args.camera)
    app.run()
