"""
Extract DeepFace + Emotion2Vec features from RAVDESS videos and cache to .npz.

Run once, then training and evaluation load from cache (instant).

Usage:
    PYTHONPATH=. venv/bin/python tests/extract_features.py

Outputs:
    data/ravdess_train_features.npz
    data/ravdess_test_features.npz
    data/elderly_features.npz
"""

import os
import sys
import time
import numpy as np
import cv2
import soundfile as sf
import tempfile
import subprocess
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    DeepFaceEmotionEmbeddingExtractor,
    Emotion2VecExtractor,
    align_face_probs,
    align_audio_probs,
    SHARED_EMOTIONS,
)
from data_split import split_dataset, RAVDESS_EMOTIONS, print_split_stats


# ============================================================================
# Configuration
# ============================================================================

RAVDESS_DIR = "data/ravdess"
ELDERLY_DIR = "/Users/Anders_1/Downloads/seedance_video_test"
OUTPUT_DIR = "data"
SAMPLE_RATE = 16000
NUM_FACE_FRAMES = 5

ELDERLY_LABEL_MAP = {
    'sad': 'Sad', 'happy': 'Happy', 'neutral': 'Neutral',
    'calm': 'Neutral', 'fear': 'Fear', 'anxious': 'Fear', 'angry': 'Angry',
}

DEEPFACE_TO_SHARED = {
    'Anger': 'Angry', 'Disgust': 'Disgust', 'Fear': 'Fear',
    'Happiness': 'Happy', 'Sadness': 'Sad', 'Surprise': 'Surprise',
    'Neutral': 'Neutral',
}


# ============================================================================
# Feature Extraction
# ============================================================================

class FeatureExtractor:
    """Extracts aligned face + audio probability vectors from videos."""

    def __init__(self):
        self.face_ext = None
        self.audio_ext = None
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def load_models(self):
        print("Loading DeepFace...")
        self.face_ext = DeepFaceEmotionEmbeddingExtractor()
        self.face_ext.load()

        print("Loading Emotion2Vec large...")
        self.audio_ext = Emotion2VecExtractor(model_size='large')
        self.audio_ext.load()
        print("Models loaded.\n")

    def extract_audio(self, video_path: str) -> Optional[np.ndarray]:
        """Extract audio track from video via ffmpeg."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        try:
            subprocess.run(
                ['ffmpeg', '-i', video_path, '-ac', '1', '-ar', str(SAMPLE_RATE),
                 '-vn', '-y', temp_path],
                capture_output=True, timeout=30)
            audio, _ = sf.read(temp_path)
            return audio.astype(np.float32)
        except Exception:
            return None
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def extract_face_probs(self, video_path: str) -> Optional[Dict[str, float]]:
        """Sample N frames, run DeepFace, average probability vectors."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None

        start = int(total_frames * 0.1)
        end = int(total_frames * 0.9)
        if end <= start:
            start, end = 0, total_frames
        indices = np.linspace(start, end - 1, NUM_FACE_FRAMES, dtype=int)

        all_probs = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) == 0:
                continue
            x, y, w, h = faces[0]
            m = int(0.1 * w)
            crop = frame[max(0, y-m):min(frame.shape[0], y+h+m),
                         max(0, x-m):min(frame.shape[1], x+w+m)]
            try:
                analysis = self.face_ext._deepface.analyze(
                    crop, actions=['emotion'],
                    detector_backend='skip',
                    enforce_detection=False, silent=True)
                raw = analysis[0]['emotion']
                probs = {
                    'Anger': raw.get('angry', 0) / 100,
                    'Disgust': raw.get('disgust', 0) / 100,
                    'Fear': raw.get('fear', 0) / 100,
                    'Happiness': raw.get('happy', 0) / 100,
                    'Sadness': raw.get('sad', 0) / 100,
                    'Surprise': raw.get('surprise', 0) / 100,
                    'Neutral': raw.get('neutral', 0) / 100,
                }
                all_probs.append(probs)
            except Exception:
                continue
        cap.release()

        if not all_probs:
            return None

        avg = {em: np.mean([p[em] for p in all_probs])
               for em in all_probs[0]}
        return avg

    def extract_audio_probs(self, audio: np.ndarray) -> Optional[Dict[str, float]]:
        """Run Emotion2Vec on audio waveform."""
        if audio is None or len(audio) < SAMPLE_RATE:
            return None
        try:
            result = self.audio_ext.extract(audio, SAMPLE_RATE)
            return result.get('emotion_probs')
        except Exception:
            return None

    def process_video(self, video_path: str) -> Optional[Dict]:
        """Extract aligned face + audio feature vectors from one video.

        Returns dict with:
            face_probs_aligned: 7-dim aligned face probabilities
            audio_probs_aligned: 7-dim aligned audio probabilities
            face_probs_raw: original DeepFace probabilities
            audio_probs_raw: original Emotion2Vec probabilities
        """
        # Audio
        audio = self.extract_audio(video_path)
        audio_probs_raw = self.extract_audio_probs(audio)

        # Face
        face_probs_raw = self.extract_face_probs(video_path)

        if face_probs_raw is None and audio_probs_raw is None:
            return None

        # Align to shared 7-class space
        face_aligned = align_face_probs(face_probs_raw) if face_probs_raw else {em: 1/7 for em in SHARED_EMOTIONS}
        audio_aligned = align_audio_probs(audio_probs_raw) if audio_probs_raw else {em: 1/7 for em in SHARED_EMOTIONS}

        # Convert to arrays in SHARED_EMOTIONS order
        face_vec = np.array([face_aligned[em] for em in SHARED_EMOTIONS], dtype=np.float32)
        audio_vec = np.array([audio_aligned[em] for em in SHARED_EMOTIONS], dtype=np.float32)

        return {
            'face_vec': face_vec,
            'audio_vec': audio_vec,
            'has_face': face_probs_raw is not None,
            'has_audio': audio_probs_raw is not None,
        }


def process_clip_list(extractor: FeatureExtractor, clips: List[Dict],
                      label_key: str = 'emotion') -> Dict:
    """Process a list of clips and return arrays for .npz storage."""
    face_vecs = []
    audio_vecs = []
    labels = []
    filenames = []
    actor_ids = []

    for i, clip in enumerate(clips):
        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - process_clip_list._start_time
            eta = (elapsed / (i + 1)) * (len(clips) - i - 1) if i > 0 else 0
            print(f"  [{i+1}/{len(clips)}] ETA: {eta:.0f}s")

        filepath = clip['filepath']
        result = extractor.process_video(filepath)

        if result is None:
            print(f"  SKIP: {clip.get('filename', filepath)} (no face or audio)")
            continue

        face_vecs.append(result['face_vec'])
        audio_vecs.append(result['audio_vec'])
        labels.append(clip[label_key])
        filenames.append(clip.get('filename', os.path.basename(filepath)))
        actor_ids.append(clip.get('actor', 0))

    # Convert labels to indices
    label_to_idx = {em: i for i, em in enumerate(SHARED_EMOTIONS)}
    label_indices = np.array([label_to_idx[l] for l in labels], dtype=np.int64)

    return {
        'face_vecs': np.array(face_vecs, dtype=np.float32),
        'audio_vecs': np.array(audio_vecs, dtype=np.float32),
        'labels': label_indices,
        'label_names': np.array(labels),
        'filenames': np.array(filenames),
        'actor_ids': np.array(actor_ids, dtype=np.int32),
        'shared_emotions': np.array(SHARED_EMOTIONS),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    extractor = FeatureExtractor()
    extractor.load_models()

    # --- RAVDESS ---
    if os.path.exists(RAVDESS_DIR):
        print("=" * 60)
        print("Processing RAVDESS dataset...")
        print("=" * 60)
        train_clips, test_clips = split_dataset(RAVDESS_DIR)
        print_split_stats(train_clips, test_clips)

        print(f"\nExtracting train features ({len(train_clips)} clips)...")
        process_clip_list._start_time = time.time()
        train_data = process_clip_list(extractor, train_clips)
        train_path = os.path.join(OUTPUT_DIR, 'ravdess_train_features.npz')
        np.savez(train_path, **train_data)
        print(f"Saved: {train_path} ({len(train_data['labels'])} samples)")

        print(f"\nExtracting test features ({len(test_clips)} clips)...")
        process_clip_list._start_time = time.time()
        test_data = process_clip_list(extractor, test_clips)
        test_path = os.path.join(OUTPUT_DIR, 'ravdess_test_features.npz')
        np.savez(test_path, **test_data)
        print(f"Saved: {test_path} ({len(test_data['labels'])} samples)")
    else:
        print(f"RAVDESS directory not found: {RAVDESS_DIR}")
        print("Download from Kaggle first: kaggle datasets download -d orvile/ravdess-dataset")

    # --- Elderly ---
    if os.path.exists(ELDERLY_DIR):
        print("\n" + "=" * 60)
        print("Processing elderly AI-generated clips...")
        print("=" * 60)

        elderly_clips = []
        for fname in sorted(os.listdir(ELDERLY_DIR)):
            if not fname.endswith('.mp4'):
                continue
            emotion_raw = fname.replace('.mp4', '').split('_')[-1]
            gt = ELDERLY_LABEL_MAP.get(emotion_raw)
            if gt is None:
                continue
            elderly_clips.append({
                'filepath': os.path.join(ELDERLY_DIR, fname),
                'filename': fname,
                'emotion': gt,
                'actor': 0,
            })

        print(f"Found {len(elderly_clips)} elderly clips")
        process_clip_list._start_time = time.time()
        elderly_data = process_clip_list(extractor, elderly_clips)
        elderly_path = os.path.join(OUTPUT_DIR, 'elderly_features.npz')
        np.savez(elderly_path, **elderly_data)
        print(f"Saved: {elderly_path} ({len(elderly_data['labels'])} samples)")
    else:
        print(f"\nElderly directory not found: {ELDERLY_DIR}")


if __name__ == "__main__":
    main()
