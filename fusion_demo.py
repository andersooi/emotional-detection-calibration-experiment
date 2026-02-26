"""
Multimodal Fusion Demo - Combined webcam + mic with real-time fusion.

Shows face-only, audio-only, and fused predictions side-by-side.
Supports toggling between V1 (Probability Fusion) and V2 (V-A Fusion).

Usage:
    python fusion_demo.py                    # Default camera (index 0)
    python fusion_demo.py --camera 1         # MacBook webcam
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

from calibration_core import (
    UserBaseline, HSEmotionExtractor, CalibrationManager,
    CalibratedDetector, average_embeddings, average_values
)
from calibration_core_audio import (
    AudioUserBaseline, Emotion2VecExtractor, AudioCalibrationManager,
    CalibratedAudioDetector, average_embeddings as avg_audio_embeddings
)
from fusion_core import MultimodalFusion, FusionResult, QUADRANT_LABELS

try:
    import sounddevice as sd
except ImportError:
    sd = None


# ============================================================================
# Configuration
# ============================================================================

SAMPLE_RATE = 16000
CHUNK_DURATION = 3.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
FRAMES_TO_AVERAGE = 25
AUDIO_STALE_THRESHOLD = 5.0  # seconds before audio is considered stale

FACE_CAL_STATES = [
    {'name': 'neutral', 'label': 'Neutral', 'duration': 5,
     'instruction': 'Look at the camera with a relaxed, natural expression.'},
    {'name': 'happy', 'label': 'Happy', 'duration': 5,
     'instruction': 'Think of a happy memory. Let yourself smile naturally.'},
    {'name': 'calm', 'label': 'Calm', 'duration': 5,
     'instruction': 'Take a deep breath. Feel relaxed and peaceful.'},
]

AUDIO_CAL_STATES = [
    {'name': 'neutral', 'label': 'Neutral', 'duration': 8,
     'instruction': 'Count from 1 to 10 in your normal voice.'},
    {'name': 'happy', 'label': 'Happy', 'duration': 10,
     'instruction': 'Tell me about something that makes you happy!'},
    {'name': 'calm', 'label': 'Calm', 'duration': 10,
     'instruction': 'Say "I feel calm and relaxed" slowly, a few times.'},
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
}


# ============================================================================
# FaceDetector (copied from calibration_demo.py)
# ============================================================================

class FaceDetector:
    """Simple face detection using OpenCV."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame: np.ndarray) -> tuple:
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
# AudioCapture (copied from calibration_demo_audio.py)
# ============================================================================

class AudioCapture:
    """Continuous audio capture from microphone."""

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
            self.buffer = self.buffer[self.chunk_samples // 2:]
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

    def get_current_buffer(self):
        return self.buffer.copy()

    def clear_buffer(self):
        self.buffer = np.array([], dtype=np.float32)
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break


# ============================================================================
# Main Application
# ============================================================================

class FusionDemoApp:
    """Main GUI for multimodal fusion testing."""

    def __init__(self, camera_index: int = 0):
        # Models
        self.face_extractor = HSEmotionExtractor()
        self.audio_extractor = Emotion2VecExtractor(model_size='large')
        self.face_detector = FaceDetector()
        self.audio_capture = AudioCapture()

        # Calibration
        self.face_cal_manager = CalibrationManager()
        self.audio_cal_manager = AudioCalibrationManager()
        self.face_cal_detector = CalibratedDetector()
        self.audio_cal_detector = CalibratedAudioDetector()

        # Fusion
        self.fusion = MultimodalFusion()

        # Camera
        self.camera_index = camera_index
        self.cap = None

        # Thread-safe latest results
        self._lock = threading.Lock()
        self._latest_face_result: Optional[Dict] = None
        self._latest_audio_result: Optional[Dict] = None
        self._face_timestamp: float = 0.0
        self._audio_timestamp: float = 0.0
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_bbox: Optional[tuple] = None

        # Timing
        self.face_latency: float = 0.0
        self.audio_latency: float = 0.0

        # State
        self.running = False
        self.current_user: Optional[str] = None

        # Calibration state
        self.cal_in_progress = False
        self.cal_modality = ''  # 'face' or 'audio'
        self.cal_state_idx = 0
        self.cal_start_time = 0.0
        self.cal_captured_frames: List[Dict] = []
        self._face_baseline: Optional[UserBaseline] = None
        self._audio_baseline: Optional[AudioUserBaseline] = None

        # Audio smoothing
        self.audio_prediction_history: List[Dict] = []
        self.audio_smoothing_window = 3

        # GUI
        self.root = tk.Tk()
        self.root.title("AIRA Multimodal Fusion Demo")
        self.root.geometry("1300x700")
        self.root.configure(bg=COLORS['bg_dark'])
        self._setup_ui()

    # ========================================================================
    # UI Setup
    # ========================================================================

    def _setup_ui(self):
        main = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main.pack(fill='both', expand=True, padx=10, pady=10)

        self._create_top_bar(main)

        content = tk.Frame(main, bg=COLORS['bg_dark'])
        content.pack(fill='both', expand=True, pady=10)

        left = tk.Frame(content, bg=COLORS['bg_dark'], width=500)
        left.pack(side='left', fill='both')
        left.pack_propagate(False)
        self._create_left_panel(left)

        right = tk.Frame(content, bg=COLORS['bg_medium'], padx=15, pady=10)
        right.pack(side='right', fill='both', expand=True, padx=(10, 0))
        self._create_right_panel(right)

        self._create_bottom_bar(main)

    def _create_top_bar(self, parent):
        bar = tk.Frame(parent, bg=COLORS['bg_dark'])
        bar.pack(fill='x', pady=(0, 5))
        tk.Label(bar, text="AIRA Multimodal Fusion", font=('Helvetica', 18, 'bold'),
                 bg=COLORS['bg_dark'], fg=COLORS['text_white']).pack(side='left')
        self.user_label = tk.Label(bar, text="[No User]", font=('Helvetica', 12),
                                   bg=COLORS['bg_dark'], fg=COLORS['text_gray'])
        self.user_label.pack(side='right')

    def _create_left_panel(self, parent):
        self.video_label = tk.Label(parent, bg='#1a1a2e')
        self.video_label.pack(pady=10)

        # Audio level
        lf = tk.Frame(parent, bg=COLORS['bg_dark'])
        lf.pack(pady=5)
        tk.Label(lf, text="Audio:", font=('Helvetica', 10), bg=COLORS['bg_dark'],
                 fg=COLORS['text_gray']).pack(side='left', padx=5)
        self.level_canvas = tk.Canvas(lf, width=200, height=15, bg=COLORS['bg_medium'],
                                      highlightthickness=0)
        self.level_canvas.pack(side='left')

        self.status_label = tk.Label(parent, text="Initializing...", font=('Helvetica', 11),
                                     bg=COLORS['bg_dark'], fg=COLORS['text_gray'])
        self.status_label.pack(pady=5)

        self.instruction_label = tk.Label(parent, text="", font=('Helvetica', 11),
                                          bg=COLORS['bg_dark'], fg=COLORS['accent_yellow'],
                                          wraplength=400, justify='center')
        self.instruction_label.pack(pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(parent, length=300, mode='determinate',
                                            variable=self.progress_var)
        self.progress_bar.pack(pady=5)
        self.progress_bar.pack_forget()

    def _create_right_panel(self, parent):
        # Three-column comparison
        cols = tk.Frame(parent, bg=COLORS['bg_medium'])
        cols.pack(fill='both', expand=True)

        self.face_widgets = self._create_prediction_column(cols, "FACE ONLY", COLORS['accent_blue'])
        self.audio_widgets = self._create_prediction_column(cols, "AUDIO ONLY", COLORS['accent_purple'])
        self.fused_widgets = self._create_prediction_column(cols, "FUSED", COLORS['accent_green'])

        # Fusion settings
        settings = tk.Frame(parent, bg=COLORS['bg_medium'])
        settings.pack(fill='x', pady=(10, 0))

        tk.Label(settings, text="FUSION:", font=('Helvetica', 10, 'bold'),
                 bg=COLORS['bg_medium'], fg=COLORS['text_white']).pack(side='left')

        self.fusion_version_var = tk.IntVar(value=1)
        tk.Radiobutton(settings, text="V1 Probability", variable=self.fusion_version_var,
                       value=1, command=self._on_version_change,
                       bg=COLORS['bg_medium'], fg=COLORS['text_white'],
                       selectcolor=COLORS['bg_dark'], activebackground=COLORS['bg_medium']
                       ).pack(side='left', padx=10)
        tk.Radiobutton(settings, text="V2 V-A Space", variable=self.fusion_version_var,
                       value=2, command=self._on_version_change,
                       bg=COLORS['bg_medium'], fg=COLORS['text_white'],
                       selectcolor=COLORS['bg_dark'], activebackground=COLORS['bg_medium']
                       ).pack(side='left')

        self.weights_label = tk.Label(settings, text="Weights: --", font=('Helvetica', 10),
                                      bg=COLORS['bg_medium'], fg=COLORS['text_gray'])
        self.weights_label.pack(side='right')

    def _create_prediction_column(self, parent, title: str, color: str) -> Dict:
        frame = tk.Frame(parent, bg=COLORS['bg_dark'], padx=8, pady=8)
        frame.pack(side='left', fill='both', expand=True, padx=3)

        tk.Label(frame, text=title, font=('Helvetica', 10, 'bold'),
                 bg=COLORS['bg_dark'], fg=color).pack()

        emotion = tk.Label(frame, text="--", font=('Helvetica', 16, 'bold'),
                           bg=COLORS['bg_dark'], fg=COLORS['text_white'])
        emotion.pack(pady=8)

        confidence = tk.Label(frame, text="Conf: --", font=('Helvetica', 9),
                              bg=COLORS['bg_dark'], fg=COLORS['text_gray'])
        confidence.pack()

        va = tk.Label(frame, text="V: -- A: --", font=('Helvetica', 9),
                      bg=COLORS['bg_dark'], fg=COLORS['text_gray'])
        va.pack()

        quadrant = tk.Label(frame, text="Q: --", font=('Helvetica', 9),
                            bg=COLORS['bg_dark'], fg=COLORS['text_gray'])
        quadrant.pack(pady=(3, 0))

        return {'emotion': emotion, 'confidence': confidence, 'va': va, 'quadrant': quadrant}

    def _create_bottom_bar(self, parent):
        bar = tk.Frame(parent, bg=COLORS['bg_dark'])
        bar.pack(fill='x', pady=(10, 0))

        btns = tk.Frame(bar, bg=COLORS['bg_dark'])
        btns.pack(side='left')

        self.cal_face_btn = tk.Button(btns, text="Calibrate Face", command=self.start_face_calibration,
                                      font=('Helvetica', 10), bg=COLORS['accent_blue'], fg='white', padx=10, pady=3)
        self.cal_face_btn.pack(side='left', padx=3)

        self.cal_audio_btn = tk.Button(btns, text="Calibrate Audio", command=self.start_audio_calibration,
                                       font=('Helvetica', 10), bg=COLORS['accent_purple'], fg='white', padx=10, pady=3)
        self.cal_audio_btn.pack(side='left', padx=3)

        self.save_btn = tk.Button(btns, text="Save Profiles", command=self.save_profiles,
                                  font=('Helvetica', 10), bg=COLORS['bg_medium'], fg='white',
                                  padx=10, pady=3, state='disabled')
        self.save_btn.pack(side='left', padx=3)

        self.metrics_label = tk.Label(bar, text="Face: -- ms | Audio: -- ms",
                                      font=('Helvetica', 10), bg=COLORS['bg_dark'], fg=COLORS['text_gray'])
        self.metrics_label.pack(side='right')

    # ========================================================================
    # Fusion Version Toggle
    # ========================================================================

    def _on_version_change(self):
        self.fusion.set_version(self.fusion_version_var.get())

    # ========================================================================
    # Calibration
    # ========================================================================

    def _ensure_user(self) -> Optional[str]:
        if self.current_user is None:
            user_id = simpledialog.askstring("User ID", "Enter a user ID:", initialvalue="test_user")
            if not user_id:
                return None
            self.current_user = user_id
            self.user_label.config(text=f"User: {user_id}")
        return self.current_user

    def start_face_calibration(self):
        if not self._ensure_user():
            return
        self.cal_in_progress = True
        self.cal_modality = 'face'
        self.cal_state_idx = 0
        self.cal_captured_frames = []
        self.cal_face_btn.config(state='disabled')
        self.cal_audio_btn.config(state='disabled')
        self._start_cal_state()

    def start_audio_calibration(self):
        if not self._ensure_user():
            return
        self.cal_in_progress = True
        self.cal_modality = 'audio'
        self.cal_state_idx = 0
        self.cal_captured_frames = []
        self.audio_capture.clear_buffer()
        self.cal_face_btn.config(state='disabled')
        self.cal_audio_btn.config(state='disabled')
        self._start_cal_state()

    def _get_cal_states(self):
        return FACE_CAL_STATES if self.cal_modality == 'face' else AUDIO_CAL_STATES

    def _start_cal_state(self):
        states = self._get_cal_states()
        if self.cal_state_idx >= len(states):
            self._complete_calibration()
            return

        state = states[self.cal_state_idx]
        modality_label = 'Face' if self.cal_modality == 'face' else 'Audio'
        self.instruction_label.config(
            text=f"Capturing {modality_label} {state['label'].upper()}\n\n{state['instruction']}"
        )
        self.progress_bar.pack(pady=5)
        self.progress_var.set(0)
        self.cal_captured_frames = []
        self.cal_start_time = time.time()

        if self.cal_modality == 'audio':
            self.audio_capture.clear_buffer()

        self.status_label.config(text=f"Calibrating {modality_label}: {state['label']} ({state['duration']}s)")

    def _process_calibration(self):
        if not self.cal_in_progress:
            return

        states = self._get_cal_states()
        state = states[self.cal_state_idx]
        elapsed = time.time() - self.cal_start_time
        self.progress_var.set(min(100, (elapsed / state['duration']) * 100))

        if elapsed >= state['duration']:
            self._finalize_cal_state()

    def _capture_face_cal_frame(self, result: Dict):
        if self.cal_in_progress and self.cal_modality == 'face':
            elapsed = time.time() - self.cal_start_time
            if elapsed > 1.0:  # Skip first second
                self.cal_captured_frames.append(result)

    def _finalize_cal_state(self):
        states = self._get_cal_states()
        state = states[self.cal_state_idx]

        if self.cal_modality == 'face':
            if len(self.cal_captured_frames) < 5:
                messagebox.showwarning("Retry", f"Not enough frames for {state['label']}.")
                self._start_cal_state()
                return

            frames = self.cal_captured_frames[-FRAMES_TO_AVERAGE:] if len(self.cal_captured_frames) >= FRAMES_TO_AVERAGE else self.cal_captured_frames
            avg_emb = average_embeddings([f['embedding'] for f in frames])
            avg_v = average_values([f['valence'] for f in frames])
            avg_a = average_values([f['arousal'] for f in frames])

            if self.cal_state_idx == 0:
                self._face_baseline = UserBaseline(user_id=self.current_user)

            name = state['name']
            if name == 'neutral':
                self._face_baseline.neutral_embedding = avg_emb
                self._face_baseline.neutral_valence = avg_v
                self._face_baseline.neutral_arousal = avg_a
            elif name == 'happy':
                self._face_baseline.happy_embedding = avg_emb
                self._face_baseline.happy_valence = avg_v
                self._face_baseline.happy_arousal = avg_a
            elif name == 'calm':
                self._face_baseline.calm_embedding = avg_emb
                self._face_baseline.calm_valence = avg_v
                self._face_baseline.calm_arousal = avg_a

        elif self.cal_modality == 'audio':
            full_audio = self.audio_capture.get_current_buffer()
            if len(full_audio) < SAMPLE_RATE * 2:
                messagebox.showwarning("Retry", f"Not enough audio for {state['label']}.")
                self._start_cal_state()
                return

            try:
                result = self.audio_extractor.extract(full_audio, SAMPLE_RATE)
                embedding = result['embedding']
            except Exception as e:
                messagebox.showerror("Error", f"Audio processing failed: {e}")
                self._start_cal_state()
                return

            if self.cal_state_idx == 0:
                self._audio_baseline = AudioUserBaseline(user_id=self.current_user)

            name = state['name']
            if name == 'neutral':
                self._audio_baseline.neutral_embedding = embedding
            elif name == 'happy':
                self._audio_baseline.happy_embedding = embedding
            elif name == 'calm':
                self._audio_baseline.calm_embedding = embedding

        self.cal_state_idx += 1
        self.cal_captured_frames = []
        if self.cal_modality == 'audio':
            self.audio_capture.clear_buffer()
        self.root.after(500, self._start_cal_state)

    def _complete_calibration(self):
        self.cal_in_progress = False
        self.progress_bar.pack_forget()
        self.instruction_label.config(text="")

        if self.cal_modality == 'face' and self._face_baseline:
            self.face_cal_detector.set_baseline(self._face_baseline)
            self.fusion.set_face_calibration(
                self._face_baseline.neutral_valence,
                self._face_baseline.neutral_arousal
            )
        elif self.cal_modality == 'audio' and self._audio_baseline:
            self.audio_cal_detector.set_baseline(self._audio_baseline)

        self.cal_face_btn.config(state='normal')
        self.cal_audio_btn.config(state='normal')
        self.save_btn.config(state='normal')
        self.status_label.config(text=f"{self.cal_modality.title()} calibration complete!")
        messagebox.showinfo("Done", f"{self.cal_modality.title()} calibration complete!")

    def save_profiles(self):
        if self._face_baseline:
            self.face_cal_manager.save_profile(self._face_baseline)
        if self._audio_baseline:
            self.audio_cal_manager.save_profile(self._audio_baseline)
        messagebox.showinfo("Saved", "Profiles saved!")

    # ========================================================================
    # Video / Audio Display
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

    def update_audio_level(self, audio_chunk):
        if audio_chunk is None or len(audio_chunk) == 0:
            return
        level = min(int(np.abs(audio_chunk).mean() * 2000), 200)
        self.level_canvas.delete("all")
        if level > 0:
            color = COLORS['accent_green'] if level < 150 else COLORS['accent_yellow']
            self.level_canvas.create_rectangle(0, 0, level, 15, fill=color, outline='')

    # ========================================================================
    # Display Update
    # ========================================================================

    def _update_column(self, widgets: Dict, emotion: str, confidence: float,
                       valence=None, arousal=None, quadrant='--', tag=''):
        widgets['emotion'].config(text=emotion)
        conf_text = f"Conf: {confidence:.0%}" if confidence > 0 else "Conf: --"
        if tag:
            conf_text += f" {tag}"
        widgets['confidence'].config(text=conf_text)

        if valence is not None and arousal is not None:
            widgets['va'].config(text=f"V: {valence:+.2f} A: {arousal:+.2f}")
        else:
            widgets['va'].config(text="V: -- A: --")

        widgets['quadrant'].config(text=f"Q: {quadrant}")

    def update_fusion_display(self):
        """Called by GUI timer at ~10Hz. Reads latest results, fuses, updates all columns."""
        if not self.running:
            return

        # Schedule next update
        self.root.after(100, self.update_fusion_display)

        # Handle calibration progress
        if self.cal_in_progress:
            self._process_calibration()

        # Read latest results (thread-safe)
        with self._lock:
            face_result = self._latest_face_result.copy() if self._latest_face_result else None
            audio_result = self._latest_audio_result.copy() if self._latest_audio_result else None
            face_age = time.time() - self._face_timestamp
            audio_age = time.time() - self._audio_timestamp
            frame = self._latest_frame.copy() if self._latest_frame is not None else None
            bbox = self._latest_bbox

        # Staleness check
        if face_result is not None and face_age > 1.0:
            face_result = None
        if audio_result is not None and audio_age > AUDIO_STALE_THRESHOLD:
            audio_result = None

        # Update video
        if frame is not None:
            self.root.after(0, lambda f=frame, b=bbox: self.update_video(f, b))

        # Update face column
        if face_result:
            self._update_column(
                self.face_widgets,
                face_result['top_emotion'],
                face_result['confidence'],
                face_result.get('valence'), face_result.get('arousal'),
                quadrant=QUADRANT_LABELS.get(
                    self._va_to_q(face_result.get('valence', 0), face_result.get('arousal', 0)), '--')
            )
        else:
            self._update_column(self.face_widgets, "NO FACE", 0)

        # Update audio column
        if audio_result:
            self._update_column(
                self.audio_widgets,
                audio_result['top_emotion'],
                audio_result['confidence'],
                quadrant='--'
            )
        else:
            self._update_column(self.audio_widgets, "NO AUDIO", 0)

        # Fuse and update fused column
        fusion_result = self.fusion.fuse(face_result, audio_result)

        modality_tag = {
            'both': '[F+A]',
            'face_only': '[F only]',
            'audio_only': '[A only]',
            'none': '[--]'
        }.get(fusion_result.modalities_present, '')

        version_tag = f"V{fusion_result.fusion_version}"

        self._update_column(
            self.fused_widgets,
            fusion_result.emotion,
            fusion_result.confidence,
            fusion_result.fused_valence, fusion_result.fused_arousal,
            quadrant=fusion_result.quadrant_label,
            tag=f"{modality_tag} {version_tag}"
        )

        # Update weights display
        if fusion_result.modalities_present == 'both':
            self.weights_label.config(
                text=f"Face: {fusion_result.face_weight:.0%} | Audio: {fusion_result.audio_weight:.0%}"
            )
        else:
            self.weights_label.config(text=f"Weights: {fusion_result.modalities_present}")

        # Update metrics
        self.metrics_label.config(
            text=f"Face: {self.face_latency*1000:.0f}ms | Audio: {self.audio_latency*1000:.0f}ms"
        )

    def _va_to_q(self, v, a):
        from fusion_core import va_to_quadrant
        return va_to_quadrant(v, a)

    # ========================================================================
    # Processing Threads
    # ========================================================================

    def face_loop(self):
        """Face processing thread."""
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

                with self._lock:
                    self._latest_frame = frame.copy()
                    self._latest_bbox = bbox

                if face_img is not None:
                    start = time.time()
                    result = self.face_extractor.extract(face_img)
                    self.face_latency = time.time() - start

                    with self._lock:
                        self._latest_face_result = result
                        self._face_timestamp = time.time()

                    # Capture for calibration if active
                    if self.cal_in_progress and self.cal_modality == 'face':
                        self.root.after(0, lambda r=result: self._capture_face_cal_frame(r))
                else:
                    with self._lock:
                        self._latest_face_result = None

                time.sleep(0.01)
            except Exception as e:
                print(f"Face loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def audio_loop(self):
        """Audio processing thread."""
        while self.running:
            try:
                # Update audio level
                buf = self.audio_capture.get_current_buffer()
                if len(buf) > 0:
                    recent = buf[-1600:] if len(buf) > 1600 else buf
                    self.root.after(0, lambda r=recent: self.update_audio_level(r))

                # Skip extraction during audio calibration (buffer is being accumulated)
                if self.cal_in_progress and self.cal_modality == 'audio':
                    time.sleep(0.1)
                    continue

                chunk = self.audio_capture.get_chunk(timeout=0.5)
                if chunk is not None and len(chunk) > 0:
                    start = time.time()
                    result = self.audio_extractor.extract(chunk, SAMPLE_RATE)
                    self.audio_latency = time.time() - start

                    with self._lock:
                        self._latest_audio_result = result
                        self._audio_timestamp = time.time()
                else:
                    time.sleep(0.05)
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
            # Load face model
            self.status_label.config(text="Loading HSEmotion model...")
            self.root.update()
            self.face_extractor.load()

            # Load audio model
            self.status_label.config(text="Loading Emotion2Vec model...")
            self.root.update()
            self.audio_extractor.load()

            # Start camera
            self.status_label.config(text=f"Starting camera {self.camera_index}...")
            self.root.update()
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # Start audio
            self.status_label.config(text="Starting audio capture...")
            self.root.update()
            try:
                self.audio_capture.start()
            except Exception as e:
                print(f"Audio start failed: {e}")

            time.sleep(0.5)
            self.running = True
            self.status_label.config(text="Ready! Calibrate or observe raw fusion.")

            # Start processing threads
            threading.Thread(target=self.face_loop, daemon=True).start()
            threading.Thread(target=self.audio_loop, daemon=True).start()

            # Start GUI update timer
            self.root.after(100, self.update_fusion_display)

        threading.Thread(target=init, daemon=True).start()
        self.root.mainloop()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='AIRA Multimodal Fusion Demo')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (0=default, 1=MacBook webcam)')
    args = parser.parse_args()

    print(f"Using camera index: {args.camera}")
    app = FusionDemoApp(camera_index=args.camera)
    app.run()
