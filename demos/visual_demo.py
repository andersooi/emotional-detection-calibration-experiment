"""
HSEmotion Calibration Test Demo

Shows raw model output vs calibrated output side-by-side.

Usage:
    python calibration_demo.py              # Use default camera (index 0)
    python calibration_demo.py --camera 1   # Use MacBook webcam (index 1)
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
    UserBaseline,
    HSEmotionExtractor,
    CalibrationManager,
    CalibratedDetector,
    average_embeddings,
    average_values
)


# ============================================================================
# Configuration
# ============================================================================

# Calibration states to capture
CALIBRATION_STATES = [
    {
        'name': 'neutral',
        'label': 'Neutral',
        'instruction': 'Please look at the camera with a relaxed, natural expression.\n'
                      'Just be comfortable - no need to smile or frown.',
        'duration': 5
    },
    {
        'name': 'happy',
        'label': 'Happy',
        'instruction': 'Think of a happy memory - perhaps a time with family or friends.\n'
                      'Let yourself smile naturally.',
        'duration': 5
    },
]

# How many frames to average for baseline (last 3-4 seconds at ~10fps)
FRAMES_TO_AVERAGE = 25

# Colors
COLORS = {
    'bg_dark': '#2C3E50',
    'bg_medium': '#34495E',
    'text_white': '#FFFFFF',
    'text_gray': '#BDC3C7',
    'accent_green': '#2ECC71',
    'accent_red': '#E74C3C',
    'accent_blue': '#3498DB',
    'accent_yellow': '#F1C40F',
    'neutral': '#95A5A6',
    'happy': '#2ECC71',
    'calm': '#3498DB',
    'sad': '#9B59B6',
    'stressed': '#E74C3C'
}


# ============================================================================
# Face Detector (OpenCV Haar Cascade)
# ============================================================================

class FaceDetector:
    """Simple face detection using OpenCV."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame: np.ndarray) -> Optional[tuple]:
        """
        Detect face in frame.

        Returns:
            (face_rgb, bbox) or (None, None) if no face found
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            return None, None

        # Get largest face
        areas = [w * h for (x, y, w, h) in faces]
        idx = np.argmax(areas)
        x, y, w, h = faces[idx]

        # Add margin
        margin = int(0.1 * w)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        # Crop and convert to RGB
        face_img = frame[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        return face_rgb, (x1, y1, x2, y2)


# ============================================================================
# Main Application
# ============================================================================

class CalibrationDemoApp:
    """Main GUI application for calibration testing."""

    def __init__(self, camera_index: int = 0):
        # Initialize components
        self.extractor = HSEmotionExtractor()
        self.face_detector = FaceDetector()
        self.calibration_manager = CalibrationManager()
        self.detector = CalibratedDetector()

        # Camera
        self.camera_index = camera_index

        # State
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.current_user: Optional[str] = None
        self.calibration_in_progress = False
        self.calibration_state_idx = 0
        self.capture_start_time = 0.0
        self.captured_frames: List[Dict] = []

        # Metrics
        self.inference_time = 0.0
        self.fps = 0.0
        self.frame_count = 0
        self.fps_start_time = time.time()

        # Setup GUI
        self.root = tk.Tk()
        self.root.title("AIRA Calibration Test - Raw vs Calibrated")
        self.root.geometry("1100x700")
        self.root.configure(bg=COLORS['bg_dark'])

        self._setup_ui()

    def _setup_ui(self):
        """Create the UI components."""
        # Main container
        main_frame = tk.Frame(self.root, bg=COLORS['bg_dark'])
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Top bar
        self._create_top_bar(main_frame)

        # Content area
        content_frame = tk.Frame(main_frame, bg=COLORS['bg_dark'])
        content_frame.pack(fill='both', expand=True, pady=10)

        # Left panel - Camera
        left_panel = tk.Frame(content_frame, bg=COLORS['bg_dark'])
        left_panel.pack(side='left', fill='both', expand=True)

        self._create_camera_panel(left_panel)

        # Right panel - Comparison
        right_panel = tk.Frame(content_frame, bg=COLORS['bg_medium'], padx=15, pady=15)
        right_panel.pack(side='right', fill='both', padx=(10, 0))

        self._create_comparison_panel(right_panel)

        # Bottom bar
        self._create_bottom_bar(main_frame)

    def _create_top_bar(self, parent):
        """Create top bar with title and user info."""
        top_bar = tk.Frame(parent, bg=COLORS['bg_dark'])
        top_bar.pack(fill='x', pady=(0, 10))

        # Title
        tk.Label(
            top_bar,
            text="AIRA Calibration Test",
            font=('Helvetica', 18, 'bold'),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_white']
        ).pack(side='left')

        # User label
        self.user_label = tk.Label(
            top_bar,
            text="[No User]",
            font=('Helvetica', 12),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_gray']
        )
        self.user_label.pack(side='right')

    def _create_camera_panel(self, parent):
        """Create camera display and status."""
        # Video label
        self.video_label = tk.Label(parent, bg='#1a1a2e')
        self.video_label.pack(pady=10)

        # Status label
        self.status_label = tk.Label(
            parent,
            text="Initializing...",
            font=('Helvetica', 11),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_gray']
        )
        self.status_label.pack(pady=5)

        # Calibration instruction (shown during calibration)
        self.instruction_label = tk.Label(
            parent,
            text="",
            font=('Helvetica', 12),
            bg=COLORS['bg_dark'],
            fg=COLORS['accent_yellow'],
            wraplength=400,
            justify='center'
        )
        self.instruction_label.pack(pady=5)

        # Calibration progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            parent,
            length=300,
            mode='determinate',
            variable=self.progress_var
        )
        self.progress_bar.pack(pady=5)
        self.progress_bar.pack_forget()  # Hide initially

    def _create_comparison_panel(self, parent):
        """Create side-by-side comparison display."""
        # Header
        tk.Label(
            parent,
            text="Prediction Comparison",
            font=('Helvetica', 14, 'bold'),
            bg=COLORS['bg_medium'],
            fg=COLORS['text_white']
        ).pack(pady=(0, 15))

        # Comparison container
        comparison_frame = tk.Frame(parent, bg=COLORS['bg_medium'])
        comparison_frame.pack(fill='both', expand=True)

        # --- Raw Output Column ---
        raw_frame = tk.Frame(comparison_frame, bg=COLORS['bg_dark'], padx=10, pady=10)
        raw_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))

        tk.Label(
            raw_frame,
            text="RAW OUTPUT",
            font=('Helvetica', 11, 'bold'),
            bg=COLORS['bg_dark'],
            fg=COLORS['accent_red']
        ).pack()

        self.raw_emotion_label = tk.Label(
            raw_frame,
            text="--",
            font=('Helvetica', 18, 'bold'),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_white']
        )
        self.raw_emotion_label.pack(pady=10)

        self.raw_confidence_label = tk.Label(
            raw_frame,
            text="Confidence: --%",
            font=('Helvetica', 10),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_gray']
        )
        self.raw_confidence_label.pack()

        tk.Label(
            raw_frame,
            text="────────────",
            font=('Helvetica', 10),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_gray']
        ).pack(pady=5)

        self.raw_va_label = tk.Label(
            raw_frame,
            text="V: -- | A: --",
            font=('Helvetica', 10),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_gray']
        )
        self.raw_va_label.pack()

        self.raw_quadrant_label = tk.Label(
            raw_frame,
            text="Quadrant: --",
            font=('Helvetica', 10),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_gray']
        )
        self.raw_quadrant_label.pack(pady=5)

        # --- Calibrated Output Column ---
        cal_frame = tk.Frame(comparison_frame, bg=COLORS['bg_dark'], padx=10, pady=10)
        cal_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))

        tk.Label(
            cal_frame,
            text="CALIBRATED OUTPUT",
            font=('Helvetica', 11, 'bold'),
            bg=COLORS['bg_dark'],
            fg=COLORS['accent_green']
        ).pack()

        self.cal_emotion_label = tk.Label(
            cal_frame,
            text="--",
            font=('Helvetica', 18, 'bold'),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_white']
        )
        self.cal_emotion_label.pack(pady=10)

        self.cal_confidence_label = tk.Label(
            cal_frame,
            text="Confidence: --%",
            font=('Helvetica', 10),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_gray']
        )
        self.cal_confidence_label.pack()

        tk.Label(
            cal_frame,
            text="────────────",
            font=('Helvetica', 10),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_gray']
        ).pack(pady=5)

        self.cal_va_label = tk.Label(
            cal_frame,
            text="V-shift: -- | A-shift: --",
            font=('Helvetica', 10),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_gray']
        )
        self.cal_va_label.pack()

        self.cal_quadrant_label = tk.Label(
            cal_frame,
            text="Quadrant: --",
            font=('Helvetica', 10),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_gray']
        )
        self.cal_quadrant_label.pack(pady=5)

        # --- Similarities Section ---
        sim_frame = tk.Frame(parent, bg=COLORS['bg_medium'])
        sim_frame.pack(fill='x', pady=(15, 0))

        tk.Label(
            sim_frame,
            text="BASELINE SIMILARITIES",
            font=('Helvetica', 10, 'bold'),
            bg=COLORS['bg_medium'],
            fg=COLORS['text_white']
        ).pack(pady=(0, 8))

        self.sim_bars = {}
        for state, color in [('neutral', '#95A5A6'), ('happy', '#2ECC71')]:
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

    def _create_bottom_bar(self, parent):
        """Create bottom bar with buttons and metrics."""
        bottom_bar = tk.Frame(parent, bg=COLORS['bg_dark'])
        bottom_bar.pack(fill='x', pady=(10, 0))

        # Buttons
        btn_frame = tk.Frame(bottom_bar, bg=COLORS['bg_dark'])
        btn_frame.pack(side='left')

        self.calibrate_btn = tk.Button(
            btn_frame,
            text="Calibrate",
            command=self.start_calibration,
            font=('Helvetica', 10),
            bg=COLORS['accent_blue'],
            fg='white',
            padx=15,
            pady=5
        )
        self.calibrate_btn.pack(side='left', padx=5)

        self.load_btn = tk.Button(
            btn_frame,
            text="Load Profile",
            command=self.load_profile,
            font=('Helvetica', 10),
            bg=COLORS['bg_medium'],
            fg='white',
            padx=15,
            pady=5
        )
        self.load_btn.pack(side='left', padx=5)

        self.save_btn = tk.Button(
            btn_frame,
            text="Save Profile",
            command=self.save_profile,
            font=('Helvetica', 10),
            bg=COLORS['bg_medium'],
            fg='white',
            padx=15,
            pady=5,
            state='disabled'
        )
        self.save_btn.pack(side='left', padx=5)

        # Metrics
        self.metrics_label = tk.Label(
            bottom_bar,
            text="Latency: -- ms | FPS: --",
            font=('Helvetica', 10),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_gray']
        )
        self.metrics_label.pack(side='right')

    # ========================================================================
    # Calibration Methods
    # ========================================================================

    def start_calibration(self):
        """Start the calibration flow."""
        # Prompt for user ID
        user_id = simpledialog.askstring(
            "User ID",
            "Enter a user ID for this calibration:",
            initialvalue="test_user"
        )

        if not user_id:
            return

        self.current_user = user_id
        self.user_label.config(text=f"User: {user_id}")

        # Reset calibration state
        self.calibration_in_progress = True
        self.calibration_state_idx = 0
        self.captured_frames = []

        # Disable buttons during calibration
        self.calibrate_btn.config(state='disabled')
        self.load_btn.config(state='disabled')

        # Start first capture
        self._start_state_capture()

    def _start_state_capture(self):
        """Start capturing a calibration state."""
        if self.calibration_state_idx >= len(CALIBRATION_STATES):
            self._complete_calibration()
            return

        state = CALIBRATION_STATES[self.calibration_state_idx]

        # Update UI
        self.instruction_label.config(
            text=f"📷 Capturing {state['label'].upper()}\n\n{state['instruction']}"
        )
        self.progress_bar.pack(pady=5)
        self.progress_var.set(0)

        self.captured_frames = []
        self.capture_start_time = time.time()

        # Status
        self.status_label.config(text=f"Calibrating: {state['label']} ({state['duration']} sec)")

    def _capture_frame_for_calibration(self, result: Dict):
        """Capture a frame during calibration."""
        if not self.calibration_in_progress or self.calibration_state_idx >= len(CALIBRATION_STATES):
            return

        state = CALIBRATION_STATES[self.calibration_state_idx]
        elapsed = time.time() - self.capture_start_time

        # Update progress
        progress = min(100, (elapsed / state['duration']) * 100)
        self.progress_var.set(progress)

        # Skip first 1 second for user to settle
        if elapsed > 1.0:
            self.captured_frames.append(result)

        # Check if capture complete
        if elapsed >= state['duration']:
            self._finalize_state_capture()

    def _finalize_state_capture(self):
        """Finalize capture for current state and move to next."""
        state = CALIBRATION_STATES[self.calibration_state_idx]

        if len(self.captured_frames) < 5:
            messagebox.showwarning(
                "Capture Failed",
                f"Not enough frames captured for {state['label']}. Please try again."
            )
            self._start_state_capture()
            return

        # Average the last N frames
        frames_to_use = self.captured_frames[-FRAMES_TO_AVERAGE:] if len(self.captured_frames) >= FRAMES_TO_AVERAGE else self.captured_frames

        embeddings = [f['embedding'] for f in frames_to_use]
        valences = [f['valence'] for f in frames_to_use]
        arousals = [f['arousal'] for f in frames_to_use]

        avg_embedding = average_embeddings(embeddings)
        avg_valence = average_values(valences)
        avg_arousal = average_values(arousals)

        # Store in temporary baseline (create if first state)
        if self.calibration_state_idx == 0:
            self._temp_baseline = UserBaseline(user_id=self.current_user)

        state_name = state['name']
        if state_name == 'neutral':
            self._temp_baseline.neutral_embedding = avg_embedding
            self._temp_baseline.neutral_valence = avg_valence
            self._temp_baseline.neutral_arousal = avg_arousal
        elif state_name == 'happy':
            self._temp_baseline.happy_embedding = avg_embedding
            self._temp_baseline.happy_valence = avg_valence
            self._temp_baseline.happy_arousal = avg_arousal
        elif state_name == 'calm':
            self._temp_baseline.calm_embedding = avg_embedding
            self._temp_baseline.calm_valence = avg_valence
            self._temp_baseline.calm_arousal = avg_arousal

        # Move to next state
        self.calibration_state_idx += 1
        self.captured_frames = []

        # Brief pause then next
        self.root.after(500, self._start_state_capture)

    def _complete_calibration(self):
        """Complete the calibration process."""
        self.calibration_in_progress = False
        self.progress_bar.pack_forget()
        self.instruction_label.config(text="")

        # Set baseline for detector
        self.detector.set_baseline(self._temp_baseline)

        # Re-enable buttons
        self.calibrate_btn.config(state='normal')
        self.load_btn.config(state='normal')
        self.save_btn.config(state='normal')

        self.status_label.config(text="Calibration complete! Now showing comparison.")

        messagebox.showinfo(
            "Calibration Complete",
            f"Calibration complete for user '{self.current_user}'.\n\n"
            "You can now see the comparison between raw and calibrated outputs.\n"
            "Click 'Save Profile' to save this calibration."
        )

    def save_profile(self):
        """Save current calibration profile."""
        if not hasattr(self, '_temp_baseline') or self._temp_baseline is None:
            messagebox.showwarning("No Calibration", "No calibration to save.")
            return

        filepath = self.calibration_manager.save_profile(self._temp_baseline)
        messagebox.showinfo("Profile Saved", f"Profile saved to:\n{filepath}")

    def load_profile(self):
        """Load an existing profile."""
        profiles = self.calibration_manager.list_profiles()

        if not profiles:
            messagebox.showinfo("No Profiles", "No saved profiles found.")
            return

        # Simple selection dialog
        user_id = simpledialog.askstring(
            "Load Profile",
            f"Available profiles: {', '.join(profiles)}\n\nEnter user ID to load:"
        )

        if not user_id:
            return

        baseline = self.calibration_manager.load_profile(user_id)
        if baseline is None:
            messagebox.showwarning("Not Found", f"Profile '{user_id}' not found.")
            return

        self.current_user = user_id
        self.user_label.config(text=f"User: {user_id}")
        self.detector.set_baseline(baseline)
        self._temp_baseline = baseline

        self.status_label.config(text=f"Loaded profile for '{user_id}'")
        self.save_btn.config(state='normal')

        messagebox.showinfo("Profile Loaded", f"Profile '{user_id}' loaded successfully.")

    # ========================================================================
    # Main Loop
    # ========================================================================

    def update_video(self, frame: np.ndarray, bbox: Optional[tuple] = None):
        """Update video display."""
        if frame is None:
            return

        # Draw bounding box
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize for display
        frame_rgb = cv2.resize(frame_rgb, (480, 360))

        # Convert to PhotoImage
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def update_comparison_display(self, raw: Dict, calibrated: Dict):
        """Update the side-by-side comparison."""
        # Raw output
        self.raw_emotion_label.config(text=raw['emotion'])
        self.raw_confidence_label.config(text=f"Confidence: {raw['confidence']:.0%}")
        self.raw_va_label.config(text=f"V: {raw['valence']:.2f} | A: {raw['arousal']:.2f}")
        self.raw_quadrant_label.config(text=f"Quadrant: {raw['quadrant_label']}")

        # Calibrated output
        if calibrated.get('calibrated', False):
            self.cal_emotion_label.config(text=calibrated['emotion'])

            # Show confidence and source
            source = calibrated.get('emotion_source', 'unknown')
            source_labels = {
                'calibration': '[CAL]',
                'va_shift': '[V-A]',
                'raw_model': '[RAW]'
            }
            source_label = source_labels.get(source, '[?]')
            self.cal_confidence_label.config(
                text=f"Confidence: {calibrated['confidence']:.0%} {source_label}"
            )

            self.cal_va_label.config(
                text=f"V-shift: {calibrated['valence_shift']:+.2f} | A-shift: {calibrated['arousal_shift']:+.2f}"
            )
            self.cal_quadrant_label.config(text=f"Quadrant: {calibrated['quadrant_label']}")

            # Similarity bars
            sims = calibrated['similarities']
            closest = calibrated['closest_baseline']

            for state in ['neutral', 'happy']:
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
            self.cal_emotion_label.config(text="[No Cal]")
            self.cal_confidence_label.config(text="Confidence: --")
            self.cal_va_label.config(text="Calibrate first")
            self.cal_quadrant_label.config(text="Quadrant: --")
            for state in ['neutral', 'happy']:
                self.sim_bars[state]['canvas'].delete('all')
                self.sim_bars[state]['label'].config(text="--")

    def show_no_face(self):
        """Show no face detected state."""
        self.raw_emotion_label.config(text="NO FACE")
        self.raw_confidence_label.config(text="Confidence: --")
        self.cal_emotion_label.config(text="NO FACE")
        self.cal_confidence_label.config(text="Confidence: --")

    def update_metrics(self):
        """Update latency and FPS display."""
        self.metrics_label.config(
            text=f"Latency: {self.inference_time*1000:.0f} ms | FPS: {self.fps:.1f}"
        )

    def main_loop(self):
        """Main processing loop (runs in thread)."""
        while self.running:
            try:
                # Capture frame
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(0.1)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                # Detect face
                face_img, bbox = self.face_detector.detect(frame)

                if face_img is not None:
                    # Run extraction
                    start_time = time.time()
                    result = self.extractor.extract(face_img)
                    self.inference_time = time.time() - start_time

                    # If calibrating, capture frame
                    if self.calibration_in_progress:
                        self.root.after(0, lambda r=result: self._capture_frame_for_calibration(r))

                    # Get raw and calibrated predictions
                    raw = self.detector.get_raw_prediction(result)
                    calibrated = self.detector.get_calibrated_prediction(result)

                    # Update UI (must be done in main thread)
                    self.root.after(0, lambda f=frame.copy(), b=bbox: self.update_video(f, b))
                    self.root.after(0, lambda r=raw, c=calibrated: self.update_comparison_display(r, c))
                else:
                    # No face
                    self.root.after(0, lambda f=frame.copy(): self.update_video(f, None))
                    self.root.after(0, self.show_no_face)

                # Update FPS
                self.frame_count += 1
                elapsed = time.time() - self.fps_start_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.fps_start_time = time.time()
                    self.root.after(0, self.update_metrics)

                time.sleep(0.01)

            except Exception as e:
                print(f"Loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def on_close(self):
        """Handle window close."""
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

    def run(self):
        """Start the application."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        def init():
            # Load model
            self.status_label.config(text="Loading HSEmotion model...")
            self.root.update()
            self.extractor.load(status_callback=lambda msg: self.root.after(0, lambda: self.status_label.config(text=msg)))

            # Start camera
            self.status_label.config(text=f"Starting camera {self.camera_index}...")
            self.root.update()
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            time.sleep(0.5)

            self.running = True
            self.status_label.config(text="Ready! Click 'Calibrate' to start.")

            # Start main loop
            main_thread = threading.Thread(target=self.main_loop, daemon=True)
            main_thread.start()

        init_thread = threading.Thread(target=init, daemon=True)
        init_thread.start()

        self.root.mainloop()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='HSEmotion Calibration Demo')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (0=iPhone/default, 1=MacBook webcam)')
    args = parser.parse_args()

    print(f"Using camera index: {args.camera}")
    app = CalibrationDemoApp(camera_index=args.camera)
    app.run()
