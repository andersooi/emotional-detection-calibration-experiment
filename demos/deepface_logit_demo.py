"""
DeepFace Logit-Space Calibration Prototype

Instead of cosine similarity on embeddings, calibrates in the model's
own decision space by correcting user-specific logit biases.

Key idea:
    b_user = z_user_neutral - z_ref_neutral  (user's neutral bias)
    z_corrected = z_live - alpha * b_user     (bias-corrected logits)
    p_corrected = softmax(z_corrected)        (corrected probabilities)

Logits are extracted via log(p + eps) from DeepFace's emotion probabilities,
which is equivalent to true pre-softmax logits up to an additive constant
(softmax is shift-invariant, so this constant cancels out).

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

FRAMES_TO_AVERAGE = 25
CALIBRATION_DURATION = 5  # seconds

# Hand-crafted reference: what DeepFace "should" output for a clearly neutral face.
# This is the anchor point for bias computation. The user's neutral face is compared
# against this to determine how much their resting face deviates from "ideal neutral."
# Order matches EMOTION_LABELS: Anger, Disgust, Fear, Happiness, Sadness, Surprise, Neutral
REFERENCE_NEUTRAL_PROBS = np.array([0.02, 0.01, 0.02, 0.03, 0.05, 0.02, 0.85])

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
# Logit Utilities
# ============================================================================

def probs_to_logits(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Convert probabilities to pseudo-logits via log transform.

    Equivalent to true pre-softmax logits up to an additive constant,
    which doesn't matter since softmax is shift-invariant.
    """
    return np.log(np.clip(probs, eps, None))


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


class LogitCalibrator:
    """
    Calibrates DeepFace by correcting user-specific logit biases.

    During calibration: captures neutral face -> computes mean logit vector.
    At inference: subtracts scaled bias from live logits -> re-applies softmax.

    The bias is the difference between the user's neutral logits and a
    hand-crafted reference neutral distribution (85% neutral, small residuals).
    """

    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        self.z_ref_neutral: Optional[np.ndarray] = None
        self.z_user_neutral: Optional[np.ndarray] = None
        self.b_user: Optional[np.ndarray] = None
        self.calibrated = False

    def set_reference(self, ref_probs: np.ndarray):
        """Set reference neutral template from probabilities."""
        self.z_ref_neutral = probs_to_logits(ref_probs)

    def calibrate(self, user_neutral_logits: np.ndarray):
        """Set user's neutral logit vector and compute bias."""
        self.z_user_neutral = user_neutral_logits
        if self.z_ref_neutral is not None:
            self.b_user = self.z_user_neutral - self.z_ref_neutral
        else:
            self.b_user = self.z_user_neutral
        self.calibrated = True

    def correct(self, live_probs: np.ndarray) -> np.ndarray:
        """Apply bias correction to live probabilities.

        Returns corrected probability distribution (7-dim).
        """
        if not self.calibrated or self.b_user is None:
            return live_probs

        z_live = probs_to_logits(live_probs)
        z_corrected = z_live - self.alpha * self.b_user
        return softmax(z_corrected)

    def get_bias_info(self) -> Dict[str, float]:
        """Get the computed bias vector as a labeled dict."""
        if self.b_user is None:
            return {}
        return {
            label: float(self.b_user[i])
            for i, label in enumerate(EMOTION_LABELS)
        }


# ============================================================================
# Face Detector
# ============================================================================

class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame):
        """Detect largest face and return BGR crop (matching DeepFace's expected input)."""
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
# DeepFace Wrapper (logit-focused)
# ============================================================================

class DeepFaceLogitExtractor:
    """Extracts emotion probabilities and logits from DeepFace."""

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
        """Extract emotion probabilities and logits from a face image."""
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

        logits = probs_to_logits(probs)
        top_idx = int(np.argmax(probs))

        return {
            'probs': probs,
            'logits': logits,
            'top_emotion': EMOTION_LABELS[top_idx],
            'confidence': float(probs[top_idx]),
        }

    def get_reference_neutral(self) -> np.ndarray:
        """Return the reference neutral probability distribution.

        Uses a hand-crafted "ideal neutral" where the neutral class dominates.
        This is the anchor for bias computation — the user's neutral face logits
        are compared against this to quantify their resting-face bias.
        """
        return REFERENCE_NEUTRAL_PROBS.copy()


# ============================================================================
# Main Application
# ============================================================================

class DeepFaceLogitDemoApp:
    """GUI for testing logit-space calibration."""

    def __init__(self, camera_index: int = 0):
        self.extractor = DeepFaceLogitExtractor()
        self.face_detector = FaceDetector()
        self.calibrator = LogitCalibrator(alpha=0.7)

        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.current_user: Optional[str] = None

        # Calibration state
        self.cal_in_progress = False
        self.cal_start_time = 0.0
        self.captured_logits: List[np.ndarray] = []

        # Probability-level EMA smoothing (smooth the signal, not just labels)
        self.raw_probs_ema: Optional[np.ndarray] = None
        self.cal_probs_ema: Optional[np.ndarray] = None
        self.ema_decay = 0.3  # weight of new frame (lower = smoother)

        # Metrics
        self.inference_time = 0.0
        self.fps = 0.0
        self.frame_count = 0
        self.fps_start_time = time.time()

        # GUI
        self.root = tk.Tk()
        self.root.title("DeepFace Logit-Space Calibration")
        self.root.geometry("1250x780")
        self.root.configure(bg=COLORS['bg_dark'])
        self._setup_ui()

    def _setup_ui(self):
        main = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main.pack(fill='both', expand=True, padx=10, pady=10)

        # Top bar
        top = tk.Frame(main, bg=COLORS['bg_dark'])
        top.pack(fill='x', pady=(0, 10))
        tk.Label(top, text="DeepFace Logit-Space Calibration",
                 font=('Helvetica', 18, 'bold'),
                 bg=COLORS['bg_dark'], fg=COLORS['text_white']).pack(side='left')
        self.user_label = tk.Label(top, text="[No User]",
                                   font=('Helvetica', 12),
                                   bg=COLORS['bg_dark'], fg=COLORS['text_gray'])
        self.user_label.pack(side='right')

        # Content: left (camera) + right (analysis)
        content = tk.Frame(main, bg=COLORS['bg_dark'])
        content.pack(fill='both', expand=True, pady=10)

        # Left — camera feed
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
        tk.Label(raw_box, text="RAW", font=('Helvetica', 11, 'bold'),
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
        tk.Label(cal_box, text="CORRECTED", font=('Helvetica', 11, 'bold'),
                 bg=COLORS['bg_dark'], fg=COLORS['accent_green']).pack()
        self.cal_emotion = tk.Label(cal_box, text="--",
                                    font=('Helvetica', 20, 'bold'),
                                    bg=COLORS['bg_dark'],
                                    fg=COLORS['text_white'])
        self.cal_emotion.pack(pady=5)
        self.cal_confidence = tk.Label(cal_box, text="--",
                                       font=('Helvetica', 10),
                                       bg=COLORS['bg_dark'],
                                       fg=COLORS['text_gray'])
        self.cal_confidence.pack()

        # Probability distribution bars
        dist = tk.Frame(right, bg=COLORS['bg_medium'])
        dist.pack(fill='both', expand=True, pady=(5, 5))

        tk.Label(dist, text="PROBABILITY DISTRIBUTIONS",
                 font=('Helvetica', 10, 'bold'),
                 bg=COLORS['bg_medium'], fg=COLORS['text_white']).pack(pady=(0, 6))

        # Header
        hdr = tk.Frame(dist, bg=COLORS['bg_medium'])
        hdr.pack(fill='x')
        tk.Label(hdr, text="", width=10,
                 bg=COLORS['bg_medium']).pack(side='left')
        tk.Label(hdr, text="Raw", font=('Helvetica', 9, 'bold'),
                 width=18, bg=COLORS['bg_medium'],
                 fg=COLORS['accent_red']).pack(side='left', padx=(8, 0))
        tk.Label(hdr, text="Corrected", font=('Helvetica', 9, 'bold'),
                 width=18, bg=COLORS['bg_medium'],
                 fg=COLORS['accent_green']).pack(side='left', padx=(8, 0))

        self.prob_bars = {}
        for label in EMOTION_LABELS:
            row = tk.Frame(dist, bg=COLORS['bg_medium'])
            row.pack(fill='x', pady=2)

            tk.Label(row, text=label[:8], font=('Helvetica', 9),
                     width=10, anchor='e',
                     bg=COLORS['bg_medium'],
                     fg=COLORS['text_gray']).pack(side='left')

            raw_c = tk.Canvas(row, width=120, height=16,
                              bg='#1e293b', highlightthickness=0)
            raw_c.pack(side='left', padx=(8, 2))
            raw_v = tk.Label(row, text="--", font=('Helvetica', 8),
                             width=5, bg=COLORS['bg_medium'],
                             fg=COLORS['text_gray'])
            raw_v.pack(side='left')

            cal_c = tk.Canvas(row, width=120, height=16,
                              bg='#1e293b', highlightthickness=0)
            cal_c.pack(side='left', padx=(8, 2))
            cal_v = tk.Label(row, text="--", font=('Helvetica', 8),
                             width=5, bg=COLORS['bg_medium'],
                             fg=COLORS['text_gray'])
            cal_v.pack(side='left')

            self.prob_bars[label] = {
                'raw_canvas': raw_c, 'raw_val': raw_v,
                'cal_canvas': cal_c, 'cal_val': cal_v,
            }

        # Bias vector display
        bias_section = tk.Frame(right, bg=COLORS['bg_medium'])
        bias_section.pack(fill='x', pady=(5, 5))
        tk.Label(bias_section, text="BIAS CORRECTION",
                 font=('Helvetica', 10, 'bold'),
                 bg=COLORS['bg_medium'], fg=COLORS['text_white']).pack(pady=(0, 3))
        self.bias_label = tk.Label(bias_section, text="[Not calibrated]",
                                   font=('Courier', 9),
                                   bg=COLORS['bg_medium'],
                                   fg=COLORS['text_gray'],
                                   justify='left', anchor='w')
        self.bias_label.pack(fill='x')

        # Alpha slider
        alpha_row = tk.Frame(right, bg=COLORS['bg_medium'])
        alpha_row.pack(fill='x', pady=(5, 0))
        tk.Label(alpha_row, text="Alpha:",
                 font=('Helvetica', 10),
                 bg=COLORS['bg_medium'],
                 fg=COLORS['text_white']).pack(side='left')
        self.alpha_var = tk.DoubleVar(value=0.7)
        self.alpha_slider = tk.Scale(
            alpha_row, from_=0.0, to=1.5, resolution=0.05,
            orient='horizontal', variable=self.alpha_var,
            command=self._on_alpha_change,
            bg=COLORS['bg_medium'], fg=COLORS['text_white'],
            highlightthickness=0, length=200)
        self.alpha_slider.pack(side='left', padx=10)
        self.alpha_label = tk.Label(alpha_row, text="0.70",
                                    font=('Helvetica', 10, 'bold'),
                                    bg=COLORS['bg_medium'],
                                    fg=COLORS['accent_blue'])
        self.alpha_label.pack(side='left')

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

    def _on_alpha_change(self, val):
        alpha = float(val)
        self.calibrator.alpha = alpha
        self.alpha_label.config(text=f"{alpha:.2f}")

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
        self.captured_logits = []
        self.cal_start_time = time.time()
        self.cal_btn.config(state='disabled')
        self.load_btn.config(state='disabled')

        self.instruction_label.config(
            text="Capturing NEUTRAL\n\n"
                 "Look at the camera with a relaxed, natural expression.\n"
                 "This captures your resting-face bias for logit correction.")
        self.progress_bar.pack(pady=5)
        self.progress_var.set(0)
        self.status_label.config(
            text=f"Calibrating: Neutral ({CALIBRATION_DURATION}s)")

    def _capture_frame_for_calibration(self, result: Dict):
        if not self.cal_in_progress:
            return
        elapsed = time.time() - self.cal_start_time
        self.progress_var.set(min(100, (elapsed / CALIBRATION_DURATION) * 100))
        if elapsed > 1.0:  # skip first second (settling)
            self.captured_logits.append(result['logits'])
        if elapsed >= CALIBRATION_DURATION:
            self._complete_calibration()

    def _complete_calibration(self):
        self.cal_in_progress = False
        self.progress_bar.pack_forget()
        self.instruction_label.config(text="")

        if len(self.captured_logits) < 5:
            messagebox.showwarning("Retry", "Not enough frames captured.")
            self.cal_btn.config(state='normal')
            self.load_btn.config(state='normal')
            return

        # Average logits from last N frames
        logits_arr = np.array(self.captured_logits[-FRAMES_TO_AVERAGE:])
        avg_logits = np.mean(logits_arr, axis=0)

        self.calibrator.calibrate(avg_logits)

        # Console diagnostics
        bias = self.calibrator.get_bias_info()
        print(f"\n[Logit Calibration] alpha={self.calibrator.alpha:.2f}  "
              f"frames={len(self.captured_logits)}")
        print(f"  User neutral probs:  "
              f"{dict(zip(EMOTION_LABELS, softmax(avg_logits)))}")
        print(f"  Bias vector (positive = reduced, negative = boosted):")
        for lbl, b in sorted(bias.items(), key=lambda x: abs(x[1]),
                              reverse=True):
            print(f"    {lbl:12s}: {b:+.3f}")

        # Update GUI bias display
        self._update_bias_display(bias)

        self.cal_btn.config(state='normal')
        self.load_btn.config(state='normal')
        self.save_btn.config(state='normal')
        self.status_label.config(text="Calibration complete!")
        messagebox.showinfo("Done",
                            f"Logit calibration complete for '{self.current_user}'.\n"
                            f"Adjust the Alpha slider to tune correction strength.")

    def _update_bias_display(self, bias: Dict[str, float]):
        lines = []
        for lbl, b in sorted(bias.items(), key=lambda x: abs(x[1]),
                              reverse=True):
            arrow = "\u2193reduced" if b > 0.01 else (
                "\u2191boosted" if b < -0.01 else "  ~zero")
            lines.append(f"  {lbl[:8]:9s}{b:+.3f} {arrow}")
        self.bias_label.config(text="\n".join(lines))

    # ========================================================================
    # Profile persistence
    # ========================================================================

    def save_profile(self):
        if not self.calibrator.calibrated:
            messagebox.showwarning("No Calibration", "Nothing to save.")
            return
        import pickle, os
        os.makedirs('./user_profiles', exist_ok=True)
        filepath = f"./user_profiles/{self.current_user}_logit.pkl"
        data = {
            'user_id': self.current_user,
            'z_user_neutral': self.calibrator.z_user_neutral,
            'z_ref_neutral': self.calibrator.z_ref_neutral,
            'alpha': self.calibrator.alpha,
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
        profiles = [f.replace('_logit.pkl', '')
                    for f in os.listdir(profiles_dir)
                    if f.endswith('_logit.pkl')]
        if not profiles:
            messagebox.showinfo("No Profiles", "No logit profiles found.")
            return
        user_id = simpledialog.askstring(
            "Load", f"Available: {', '.join(profiles)}\n\nEnter user ID:")
        if not user_id:
            return
        filepath = f"{profiles_dir}/{user_id}_logit.pkl"
        if not os.path.exists(filepath):
            messagebox.showwarning("Not Found",
                                   f"Profile '{user_id}' not found.")
            return
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.current_user = user_id
        self.user_label.config(text=f"User: {user_id}")
        self.calibrator.z_ref_neutral = data['z_ref_neutral']
        self.calibrator.calibrate(data['z_user_neutral'])
        self.calibrator.alpha = data.get('alpha', 0.7)
        self.alpha_var.set(self.calibrator.alpha)
        self.alpha_label.config(text=f"{self.calibrator.alpha:.2f}")
        self.save_btn.config(state='normal')
        self.status_label.config(
            text=f"Loaded logit profile for '{user_id}'")
        self._update_bias_display(self.calibrator.get_bias_info())

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

    def _ema_smooth(self, new_probs: np.ndarray, ema_attr: str) -> np.ndarray:
        """Exponential moving average on probability vectors.

        Smooths the distribution itself, not just the argmax label.
        This gives an honest view of the correction's stability.
        """
        prev = getattr(self, ema_attr)
        if prev is None:
            smoothed = new_probs.copy()
        else:
            smoothed = self.ema_decay * new_probs + (1 - self.ema_decay) * prev
        setattr(self, ema_attr, smoothed)
        return smoothed

    def update_display(self, result: Dict):
        raw_probs = result['probs']

        # EMA-smooth raw probabilities
        raw_smooth = self._ema_smooth(raw_probs, 'raw_probs_ema')
        raw_top_idx = int(np.argmax(raw_smooth))
        raw_top = EMOTION_LABELS[raw_top_idx]
        raw_conf = float(raw_smooth[raw_top_idx])

        # Corrected output (correction applied to per-frame probs, then smoothed)
        corrected_probs = self.calibrator.correct(raw_probs)
        cal_smooth = self._ema_smooth(corrected_probs, 'cal_probs_ema')
        cal_top_idx = int(np.argmax(cal_smooth))
        cal_top = EMOTION_LABELS[cal_top_idx]
        cal_conf = float(cal_smooth[cal_top_idx])

        self.raw_emotion.config(text=raw_top)
        self.raw_confidence.config(text=f"{raw_conf:.0%}")

        if self.calibrator.calibrated:
            self.cal_emotion.config(text=cal_top)
            self.cal_confidence.config(text=f"{cal_conf:.0%}")
        else:
            self.cal_emotion.config(text="[No Cal]")
            self.cal_confidence.config(text="Calibrate first")

        # Probability bars (show smoothed distributions)
        for i, label in enumerate(EMOTION_LABELS):
            bars = self.prob_bars[label]
            color = EMOTION_COLORS[label]

            # Raw (smoothed)
            rp = float(raw_smooth[i])
            bars['raw_canvas'].delete('all')
            w = max(0, min(120, int(120 * rp)))
            if w > 0:
                bars['raw_canvas'].create_rectangle(
                    0, 0, w, 16, fill=color, outline='')
            bars['raw_val'].config(text=f"{rp:.0%}")

            # Corrected (smoothed)
            cp = float(cal_smooth[i])
            bars['cal_canvas'].delete('all')
            w = max(0, min(120, int(120 * cp)))
            if w > 0:
                fill = color if self.calibrator.calibrated else '#475569'
                bars['cal_canvas'].create_rectangle(
                    0, 0, w, 16, fill=fill, outline='')
            bars['cal_val'].config(
                text=f"{cp:.0%}" if self.calibrator.calibrated else "--")

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

            # Set reference neutral template (hand-crafted ideal neutral)
            ref_probs = self.extractor.get_reference_neutral()
            self.calibrator.set_reference(ref_probs)
            print(f"[Reference Neutral] {dict(zip(EMOTION_LABELS, [f'{p:.3f}' for p in ref_probs]))}")

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
        description='DeepFace Logit-Space Calibration Demo')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (0=default, 1=MacBook webcam)')
    args = parser.parse_args()

    print(f"Using camera index: {args.camera}")
    app = DeepFaceLogitDemoApp(camera_index=args.camera)
    app.run()
