"""
Emotion2Vec Audio Calibration Test Demo

Shows raw model output vs calibrated output side-by-side for audio.

Usage:
    python calibration_demo_audio.py
"""

import time
import threading
import queue
import numpy as np
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from typing import Optional, List, Dict

from calibration_core_audio import (
    AudioUserBaseline,
    Emotion2VecExtractor,
    AudioCalibrationManager,
    CalibratedAudioDetector,
    average_embeddings
)

# Try to import audio libraries
try:
    import sounddevice as sd
    AUDIO_BACKEND = 'sounddevice'
except ImportError:
    sd = None
    AUDIO_BACKEND = None


# ============================================================================
# Configuration
# ============================================================================

# Audio settings
SAMPLE_RATE = 16000  # Emotion2Vec expects 16kHz
CHUNK_DURATION = 3.0  # Process 3 seconds at a time
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

# Calibration states to capture
CALIBRATION_STATES = [
    {
        'name': 'neutral',
        'label': 'Neutral',
        'instruction': 'Please count from 1 to 10 in your normal, relaxed voice.\n'
                      'Speak naturally as if talking to a friend.',
        'duration': 8
    },
    {
        'name': 'happy',
        'label': 'Happy',
        'instruction': 'Tell me about something that makes you happy!\n'
                      'It could be a hobby, a pet, or a fond memory.\n'
                      'Let your joy come through in your voice.',
        'duration': 10
    },
    {
        'name': 'calm',
        'label': 'Calm',
        'instruction': 'Take a deep breath and speak slowly.\n'
                      'Say: "I feel calm, relaxed, and at peace."\n'
                      'Repeat it a few times in a soothing tone.',
        'duration': 10
    }
]

# Colors (same as visual demo)
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
# Audio Capture
# ============================================================================

class AudioCapture:
    """Continuous audio capture from microphone."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, chunk_duration: float = CHUNK_DURATION):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)

        self.running = False
        self.audio_queue = queue.Queue()
        self.buffer = np.array([], dtype=np.float32)
        self.stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice stream."""
        if status:
            print(f"Audio status: {status}")

        # Add to buffer
        audio_chunk = indata[:, 0].copy()  # Mono
        self.buffer = np.concatenate([self.buffer, audio_chunk])

        # Debug: print buffer size occasionally
        if len(self.buffer) % 16000 < 1600:  # Every ~1 second
            print(f"  Audio buffer: {len(self.buffer)} samples, max={np.abs(audio_chunk).max():.3f}")

        # If buffer has enough samples, put in queue
        while len(self.buffer) >= self.chunk_samples:
            chunk = self.buffer[:self.chunk_samples]
            self.buffer = self.buffer[self.chunk_samples // 2:]  # 50% overlap
            self.audio_queue.put(chunk)

    def start(self):
        """Start audio capture."""
        if sd is None:
            raise RuntimeError("sounddevice not installed. Run: pip install sounddevice")

        self.running = True
        self.buffer = np.array([], dtype=np.float32)

        # Show available devices
        print(f"Default input device: {sd.default.device[0]}")
        print(f"Starting audio stream at {self.sample_rate}Hz...")

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=self._audio_callback,
            blocksize=int(self.sample_rate * 0.1)  # 100ms blocks
        )
        self.stream.start()
        print("Audio stream started!")

    def stop(self):
        """Stop audio capture."""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get next audio chunk from queue."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_current_buffer(self) -> np.ndarray:
        """Get current audio buffer (for calibration capture)."""
        return self.buffer.copy()

    def clear_buffer(self):
        """Clear the audio buffer."""
        self.buffer = np.array([], dtype=np.float32)
        # Clear queue too
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break


# ============================================================================
# Main Application
# ============================================================================

class AudioCalibrationDemoApp:
    """Main GUI application for audio calibration testing."""

    def __init__(self):
        # Initialize components
        self.extractor = Emotion2VecExtractor(model_size='large')
        self.calibration_manager = AudioCalibrationManager()
        self.detector = CalibratedAudioDetector()
        self.audio_capture = AudioCapture()

        # State
        self.running = False
        self.current_user: Optional[str] = None
        self.calibration_in_progress = False
        self.calibration_state_idx = 0
        self.capture_start_time = 0.0
        self.captured_audio: List[np.ndarray] = []

        # Prediction smoothing (to reduce jumping between emotions)
        self.prediction_history: List[Dict] = []
        self.smoothing_window = 3  # Number of predictions to average
        self.current_smoothed_emotion = "Neutral"
        self.emotion_change_threshold = 2  # Need N consecutive different predictions to change

        # Metrics
        self.inference_time = 0.0
        self.chunks_processed = 0

        # Setup GUI
        self.root = tk.Tk()
        self.root.title("AIRA Audio Calibration Test - Raw vs Calibrated")
        self.root.geometry("900x600")
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

        # Left panel - Audio visualization / Status
        left_panel = tk.Frame(content_frame, bg=COLORS['bg_dark'])
        left_panel.pack(side='left', fill='both', expand=True)

        self._create_audio_panel(left_panel)

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

        tk.Label(
            top_bar,
            text="AIRA Audio Calibration Test",
            font=('Helvetica', 18, 'bold'),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_white']
        ).pack(side='left')

        self.user_label = tk.Label(
            top_bar,
            text="[No User]",
            font=('Helvetica', 12),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_gray']
        )
        self.user_label.pack(side='right')

    def _create_audio_panel(self, parent):
        """Create audio status and visualization panel."""
        # Status label
        self.status_label = tk.Label(
            parent,
            text="Initializing...",
            font=('Helvetica', 14),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_gray']
        )
        self.status_label.pack(pady=20)

        # Audio level indicator (simple bar)
        level_frame = tk.Frame(parent, bg=COLORS['bg_dark'])
        level_frame.pack(pady=10)

        tk.Label(
            level_frame,
            text="Audio Level:",
            font=('Helvetica', 10),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_gray']
        ).pack(side='left', padx=5)

        self.level_canvas = tk.Canvas(
            level_frame,
            width=200,
            height=20,
            bg=COLORS['bg_medium'],
            highlightthickness=1,
            highlightbackground='#7F8C8D'
        )
        self.level_canvas.pack(side='left', padx=5)

        # Instruction label (for calibration)
        self.instruction_label = tk.Label(
            parent,
            text="",
            font=('Helvetica', 12),
            bg=COLORS['bg_dark'],
            fg=COLORS['accent_yellow'],
            wraplength=400,
            justify='center'
        )
        self.instruction_label.pack(pady=20)

        # Progress bar (for calibration)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            parent,
            length=300,
            mode='determinate',
            variable=self.progress_var
        )
        self.progress_bar.pack(pady=5)
        self.progress_bar.pack_forget()

    def _create_comparison_panel(self, parent):
        """Create side-by-side comparison display."""
        tk.Label(
            parent,
            text="Prediction Comparison",
            font=('Helvetica', 14, 'bold'),
            bg=COLORS['bg_medium'],
            fg=COLORS['text_white']
        ).pack(pady=(0, 15))

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

        # --- Similarities Section ---
        sim_frame = tk.Frame(parent, bg=COLORS['bg_medium'])
        sim_frame.pack(fill='x', pady=(15, 0))

        tk.Label(
            sim_frame,
            text="BASELINE SIMILARITIES",
            font=('Helvetica', 10, 'bold'),
            bg=COLORS['bg_medium'],
            fg=COLORS['text_white']
        ).pack()

        self.similarities_label = tk.Label(
            sim_frame,
            text="neutral: -- | happy: -- | calm: --",
            font=('Helvetica', 10),
            bg=COLORS['bg_medium'],
            fg=COLORS['text_gray']
        )
        self.similarities_label.pack(pady=5)

    def _create_bottom_bar(self, parent):
        """Create bottom bar with buttons and metrics."""
        bottom_bar = tk.Frame(parent, bg=COLORS['bg_dark'])
        bottom_bar.pack(fill='x', pady=(10, 0))

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

        self.metrics_label = tk.Label(
            bottom_bar,
            text="Latency: -- ms",
            font=('Helvetica', 10),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_gray']
        )
        self.metrics_label.pack(side='right')

    # ========================================================================
    # Audio Level Visualization
    # ========================================================================

    def update_audio_level(self, audio_chunk: np.ndarray):
        """Update audio level indicator."""
        if audio_chunk is None or len(audio_chunk) == 0:
            level = 0
        else:
            level = np.abs(audio_chunk).mean()

        # Map to bar width (0-200)
        bar_width = min(int(level * 2000), 200)

        self.level_canvas.delete("all")
        if bar_width > 0:
            color = COLORS['accent_green'] if bar_width < 150 else COLORS['accent_yellow']
            self.level_canvas.create_rectangle(0, 0, bar_width, 20, fill=color, outline='')

    # ========================================================================
    # Calibration Methods
    # ========================================================================

    def start_calibration(self):
        """Start the calibration flow."""
        user_id = simpledialog.askstring(
            "User ID",
            "Enter a user ID for this audio calibration:",
            initialvalue="test_user"
        )

        if not user_id:
            return

        self.current_user = user_id
        self.user_label.config(text=f"User: {user_id}")

        self.calibration_in_progress = True
        self.calibration_state_idx = 0
        self.captured_audio = []

        self.calibrate_btn.config(state='disabled')
        self.load_btn.config(state='disabled')

        # Clear audio buffer
        self.audio_capture.clear_buffer()

        self._start_state_capture()

    def _start_state_capture(self):
        """Start capturing a calibration state."""
        if self.calibration_state_idx >= len(CALIBRATION_STATES):
            self._complete_calibration()
            return

        state = CALIBRATION_STATES[self.calibration_state_idx]

        self.instruction_label.config(
            text=f"🎤 Recording {state['label'].upper()}\n\n{state['instruction']}"
        )
        self.progress_bar.pack(pady=5)
        self.progress_var.set(0)

        self.captured_audio = []
        self.capture_start_time = time.time()
        self.audio_capture.clear_buffer()

        self.status_label.config(text=f"Recording: {state['label']} ({state['duration']} sec)")

    def _process_calibration_audio(self):
        """Process audio during calibration."""
        if not self.calibration_in_progress:
            return

        state = CALIBRATION_STATES[self.calibration_state_idx]
        elapsed = time.time() - self.capture_start_time

        progress = min(100, (elapsed / state['duration']) * 100)
        self.progress_var.set(progress)

        if elapsed >= state['duration']:
            self._finalize_state_capture()

    def _finalize_state_capture(self):
        """Finalize capture for current state and move to next."""
        state = CALIBRATION_STATES[self.calibration_state_idx]

        # Get all captured audio
        full_audio = self.audio_capture.get_current_buffer()

        if len(full_audio) < SAMPLE_RATE * 2:  # At least 2 seconds
            messagebox.showwarning(
                "Capture Failed",
                f"Not enough audio captured for {state['label']}. Please try again."
            )
            self._start_state_capture()
            return

        # Extract embedding from the full audio
        try:
            result = self.extractor.extract(full_audio, SAMPLE_RATE)
            embedding = result['embedding']
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process audio: {e}")
            self._start_state_capture()
            return

        # Store in temporary baseline
        if self.calibration_state_idx == 0:
            self._temp_baseline = AudioUserBaseline(user_id=self.current_user)

        state_name = state['name']
        if state_name == 'neutral':
            self._temp_baseline.neutral_embedding = embedding
        elif state_name == 'happy':
            self._temp_baseline.happy_embedding = embedding
        elif state_name == 'calm':
            self._temp_baseline.calm_embedding = embedding

        # Move to next state
        self.calibration_state_idx += 1
        self.audio_capture.clear_buffer()

        self.root.after(1000, self._start_state_capture)

    def _complete_calibration(self):
        """Complete the calibration process."""
        self.calibration_in_progress = False
        self.progress_bar.pack_forget()
        self.instruction_label.config(text="")

        self.detector.set_baseline(self._temp_baseline)

        self.calibrate_btn.config(state='normal')
        self.load_btn.config(state='normal')
        self.save_btn.config(state='normal')

        self.status_label.config(text="Calibration complete! Now showing comparison.")

        messagebox.showinfo(
            "Calibration Complete",
            f"Audio calibration complete for user '{self.current_user}'.\n\n"
            "You can now see the comparison between raw and calibrated outputs.\n"
            "Click 'Save Profile' to save this calibration."
        )

    def save_profile(self):
        """Save current calibration profile."""
        if not hasattr(self, '_temp_baseline') or self._temp_baseline is None:
            messagebox.showwarning("No Calibration", "No calibration to save.")
            return

        filepath = self.calibration_manager.save_profile(self._temp_baseline)
        messagebox.showinfo("Profile Saved", f"Audio profile saved to:\n{filepath}")

    def load_profile(self):
        """Load an existing profile."""
        profiles = self.calibration_manager.list_profiles()

        if not profiles:
            messagebox.showinfo("No Profiles", "No saved audio profiles found.")
            return

        user_id = simpledialog.askstring(
            "Load Profile",
            f"Available audio profiles: {', '.join(profiles)}\n\nEnter user ID to load:"
        )

        if not user_id:
            return

        baseline = self.calibration_manager.load_profile(user_id)
        if baseline is None:
            messagebox.showwarning("Not Found", f"Audio profile '{user_id}' not found.")
            return

        self.current_user = user_id
        self.user_label.config(text=f"User: {user_id}")
        self.detector.set_baseline(baseline)
        self._temp_baseline = baseline

        self.status_label.config(text=f"Loaded audio profile for '{user_id}'")
        self.save_btn.config(state='normal')

        messagebox.showinfo("Profile Loaded", f"Audio profile '{user_id}' loaded successfully.")

    # ========================================================================
    # Prediction Smoothing
    # ========================================================================

    def get_smoothed_prediction(self, calibrated: Dict) -> Dict:
        """
        Smooth predictions over a window to reduce jumping.

        Uses majority voting over recent predictions with hysteresis.
        """
        # Add to history
        self.prediction_history.append({
            'emotion': calibrated.get('emotion', 'Neutral'),
            'confidence': calibrated.get('confidence', 0.0),
            'emotion_source': calibrated.get('emotion_source', 'unknown')
        })

        # Keep only recent history
        if len(self.prediction_history) > self.smoothing_window:
            self.prediction_history = self.prediction_history[-self.smoothing_window:]

        # Count emotions in history
        emotion_counts = {}
        emotion_confidences = {}
        for pred in self.prediction_history:
            em = pred['emotion']
            emotion_counts[em] = emotion_counts.get(em, 0) + 1
            if em not in emotion_confidences:
                emotion_confidences[em] = []
            emotion_confidences[em].append(pred['confidence'])

        # Find most common emotion
        most_common = max(emotion_counts, key=emotion_counts.get)
        most_common_count = emotion_counts[most_common]

        # Hysteresis: only change if new emotion appears enough times
        if most_common != self.current_smoothed_emotion:
            if most_common_count >= self.emotion_change_threshold:
                self.current_smoothed_emotion = most_common

        # Average confidence for the smoothed emotion
        if self.current_smoothed_emotion in emotion_confidences:
            avg_confidence = sum(emotion_confidences[self.current_smoothed_emotion]) / len(emotion_confidences[self.current_smoothed_emotion])
        else:
            avg_confidence = calibrated.get('confidence', 0.0)

        # Get source from most recent prediction with this emotion
        source = 'unknown'
        for pred in reversed(self.prediction_history):
            if pred['emotion'] == self.current_smoothed_emotion:
                source = pred['emotion_source']
                break

        return {
            'emotion': self.current_smoothed_emotion,
            'confidence': avg_confidence,
            'emotion_source': source,
            'raw_emotion': calibrated.get('emotion', 'Neutral'),  # Original unsmoothed
            'history_size': len(self.prediction_history)
        }

    # ========================================================================
    # Main Loop
    # ========================================================================

    def update_comparison_display(self, raw: Dict, calibrated: Dict, smoothed: Dict = None):
        """Update the side-by-side comparison."""
        self.raw_emotion_label.config(text=raw['emotion'])
        self.raw_confidence_label.config(text=f"Confidence: {raw['confidence']:.0%}")

        if calibrated.get('calibrated', False):
            # Use smoothed prediction if available, otherwise use raw calibrated
            display = smoothed if smoothed else calibrated

            self.cal_emotion_label.config(text=display['emotion'])

            source = display.get('emotion_source', 'unknown')
            source_labels = {
                'calibration': '[CAL]',
                'raw_model': '[RAW]'
            }
            source_label = source_labels.get(source, '[?]')

            # Show if smoothed
            smooth_indicator = " ~" if smoothed else ""
            self.cal_confidence_label.config(
                text=f"Confidence: {display['confidence']:.0%} {source_label}{smooth_indicator}"
            )

            sims = calibrated['similarities']
            closest = calibrated['closest_baseline']

            sim_text = ""
            for state in ['neutral', 'happy', 'calm']:
                marker = "*" if state == closest else ""
                sim_text += f"{state}: {sims[state]:.2f}{marker}  "

            self.similarities_label.config(text=sim_text)
        else:
            self.cal_emotion_label.config(text="[No Cal]")
            self.cal_confidence_label.config(text="Confidence: --")
            self.similarities_label.config(text="neutral: -- | happy: -- | calm: --")

    def show_no_audio(self):
        """Show no audio state."""
        self.raw_emotion_label.config(text="NO AUDIO")
        self.raw_confidence_label.config(text="Confidence: --")
        self.cal_emotion_label.config(text="NO AUDIO")
        self.cal_confidence_label.config(text="Confidence: --")

    def main_loop(self):
        """Main processing loop (runs in thread)."""
        while self.running:
            try:
                # Always update audio level (even during calibration)
                current_buffer = self.audio_capture.get_current_buffer()
                if len(current_buffer) > 0:
                    # Show level from recent audio
                    recent = current_buffer[-1600:] if len(current_buffer) > 1600 else current_buffer
                    self.root.after(0, lambda r=recent: self.update_audio_level(r))

                # Handle calibration
                if self.calibration_in_progress:
                    self.root.after(0, self._process_calibration_audio)
                    time.sleep(0.1)
                    continue

                # Get audio chunk
                audio_chunk = self.audio_capture.get_chunk(timeout=0.1)

                if audio_chunk is not None and len(audio_chunk) > 0:
                    # Update level meter
                    self.root.after(0, lambda c=audio_chunk: self.update_audio_level(c))

                    # Run extraction
                    start_time = time.time()
                    result = self.extractor.extract(audio_chunk, SAMPLE_RATE)
                    self.inference_time = time.time() - start_time

                    # Get raw and calibrated predictions
                    raw = self.detector.get_raw_prediction(result)
                    calibrated = self.detector.get_calibrated_prediction(result)

                    # Apply smoothing to reduce prediction jumping
                    smoothed = self.get_smoothed_prediction(calibrated)

                    # Update UI
                    self.root.after(0, lambda r=raw, c=calibrated, s=smoothed: self.update_comparison_display(r, c, s))
                    self.root.after(0, lambda: self.metrics_label.config(
                        text=f"Latency: {self.inference_time*1000:.0f} ms"
                    ))

                    self.chunks_processed += 1

                time.sleep(0.05)

            except Exception as e:
                print(f"Loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)

    def on_close(self):
        """Handle window close."""
        self.running = False
        self.audio_capture.stop()
        self.root.destroy()

    def run(self):
        """Start the application."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        def init():
            self.status_label.config(text="Loading Emotion2Vec model...")
            self.root.update()

            try:
                self.extractor.load(
                    status_callback=lambda msg: self.root.after(0, lambda: self.status_label.config(text=msg))
                )
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load model: {e}"))
                return

            self.status_label.config(text="Starting audio capture...")
            self.root.update()

            try:
                self.audio_capture.start()
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to start audio: {e}"))
                return

            time.sleep(0.5)
            self.running = True
            self.status_label.config(text="Ready! Click 'Calibrate' to start.")

            main_thread = threading.Thread(target=self.main_loop, daemon=True)
            main_thread.start()

        init_thread = threading.Thread(target=init, daemon=True)
        init_thread.start()

        self.root.mainloop()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    if AUDIO_BACKEND is None:
        print("ERROR: sounddevice not installed.")
        print("Install with: pip install sounddevice")
        exit(1)

    app = AudioCalibrationDemoApp()
    app.run()
