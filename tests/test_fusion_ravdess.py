"""
Evaluate DeepFace + Emotion2Vec fusion on RAVDESS test videos.

Compares face-only, audio-only, and fused accuracy (no calibration).
Uses the same ProbabilityFusion logic as the live demo.

Usage:
    PYTHONPATH=. venv/bin/python tests/test_fusion_ravdess.py

Outputs per-emotion accuracy table and overall accuracy for each approach.
"""

import os
import sys
import csv
import time
import numpy as np
import cv2
import soundfile as sf
from collections import defaultdict
from typing import Dict, Optional, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    DeepFaceEmotionEmbeddingExtractor,
    Emotion2VecExtractor,
    ProbabilityFusion,
    align_face_probs,
    align_audio_probs,
    SHARED_EMOTIONS,
)


# ============================================================================
# Configuration
# ============================================================================

VIDEO_DIR = "/Users/Anders_1/Desktop/Courses/NUS/Year4/Sem2/BT4103/MAE-DFER/test_videos"
SAMPLE_RATE = 16000
NUM_FACE_FRAMES = 5  # Evenly-spaced frames to sample per video

# RAVDESS emotion codes → shared emotion labels
RAVDESS_EMOTIONS = {
    '01': 'Neutral',
    '02': 'Neutral',   # Calm → Neutral (no Calm in shared emotions)
    '03': 'Happy',
    '04': 'Sad',
    '05': 'Angry',
    '06': 'Fear',
    '07': 'Disgust',
    '08': 'Surprise',
}

# DeepFace labels → shared labels (same mapping as fusion.py)
DEEPFACE_TO_SHARED = {
    'Anger': 'Angry',
    'Disgust': 'Disgust',
    'Fear': 'Fear',
    'Happiness': 'Happy',
    'Sadness': 'Sad',
    'Surprise': 'Surprise',
    'Neutral': 'Neutral',
}

EMOTION_LABELS_DEEPFACE = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']


# ============================================================================
# Face Detection
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
            return None
        areas = [w * h for (x, y, w, h) in faces]
        idx = np.argmax(areas)
        x, y, w, h = faces[idx]
        margin = int(0.1 * w)
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)
        return frame[y1:y2, x1:x2].copy()


# ============================================================================
# Video Processing
# ============================================================================

def extract_audio_from_video(video_path: str) -> Optional[np.ndarray]:
    """Extract audio track from video file using ffmpeg via temp file."""
    import tempfile
    import subprocess

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name

    try:
        subprocess.run(
            ['ffmpeg', '-i', video_path, '-ac', '1', '-ar', str(SAMPLE_RATE),
             '-vn', '-y', temp_path],
            capture_output=True, timeout=30)
        audio, sr = sf.read(temp_path)
        return audio.astype(np.float32)
    except Exception as e:
        print(f"  Audio extraction failed: {e}")
        return None
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def sample_face_frames(video_path: str, n_frames: int = NUM_FACE_FRAMES) -> List[np.ndarray]:
    """Sample N evenly-spaced BGR face crops from a video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # Skip first/last 10% to avoid title cards or fade-outs
    start = int(total_frames * 0.1)
    end = int(total_frames * 0.9)
    if end <= start:
        start, end = 0, total_frames

    indices = np.linspace(start, end - 1, n_frames, dtype=int)
    detector = FaceDetector()
    faces = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        face_bgr = detector.detect(frame)
        if face_bgr is not None:
            faces.append(face_bgr)

    cap.release()
    return faces


def get_deepface_probs(face_extractor, face_bgr) -> Dict[str, float]:
    """Get emotion probs from DeepFace full pipeline on a BGR face crop."""
    analysis = face_extractor._deepface.analyze(
        face_bgr, actions=['emotion'],
        detector_backend='skip',
        enforce_detection=False,
        silent=True,
    )
    raw = analysis[0]['emotion']
    return {
        'Anger': raw.get('angry', 0.0) / 100,
        'Disgust': raw.get('disgust', 0.0) / 100,
        'Fear': raw.get('fear', 0.0) / 100,
        'Happiness': raw.get('happy', 0.0) / 100,
        'Sadness': raw.get('sad', 0.0) / 100,
        'Surprise': raw.get('surprise', 0.0) / 100,
        'Neutral': raw.get('neutral', 0.0) / 100,
    }


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    print("=" * 70)
    print("RAVDESS Fusion Evaluation: Face-only vs Audio-only vs Fused")
    print("=" * 70)

    # List valid video files
    videos = []
    for fname in sorted(os.listdir(VIDEO_DIR)):
        if not fname.endswith('.mp4'):
            continue
        parts = fname.replace('.mp4', '').split('-')
        if len(parts) < 7:
            continue
        emotion_code = parts[2]
        if emotion_code not in RAVDESS_EMOTIONS:
            continue
        videos.append((os.path.join(VIDEO_DIR, fname), fname, RAVDESS_EMOTIONS[emotion_code]))

    print(f"\nFound {len(videos)} valid videos")
    emotion_counts = defaultdict(int)
    for _, _, gt in videos:
        emotion_counts[gt] += 1
    for em, count in sorted(emotion_counts.items()):
        print(f"  {em}: {count}")

    # Load models
    print("\nLoading DeepFace...")
    face_extractor = DeepFaceEmotionEmbeddingExtractor()
    face_extractor.load()

    print("Loading Emotion2Vec large...")
    audio_extractor = Emotion2VecExtractor(model_size='large')
    audio_extractor.load()

    fusion = ProbabilityFusion()

    # Process videos
    results = []
    start_time = time.time()

    for i, (video_path, fname, ground_truth) in enumerate(videos):
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (len(videos) - i - 1) if i > 0 else 0
            print(f"\n[{i+1}/{len(videos)}] ETA: {eta:.0f}s")

        # --- Audio ---
        audio = extract_audio_from_video(video_path)
        audio_result = None
        audio_pred = None
        if audio is not None and len(audio) > SAMPLE_RATE:  # at least 1s
            try:
                audio_result = audio_extractor.extract(audio, SAMPLE_RATE)
                audio_pred = audio_result.get('top_emotion')
            except Exception as e:
                print(f"  Audio extract failed: {e}")

        # --- Face ---
        face_crops = sample_face_frames(video_path)
        face_result = None
        face_pred = None
        if face_crops:
            # Average probability vectors across sampled frames
            all_probs = []
            for crop in face_crops:
                try:
                    probs = get_deepface_probs(face_extractor, crop)
                    all_probs.append(probs)
                except Exception:
                    continue

            if all_probs:
                avg_probs = {}
                for em in EMOTION_LABELS_DEEPFACE:
                    avg_probs[em] = np.mean([p[em] for p in all_probs])
                face_top = max(avg_probs, key=avg_probs.get)
                face_result = {
                    'top_emotion': face_top,
                    'confidence': avg_probs[face_top],
                    'emotion_probs': avg_probs,
                }
                face_pred = DEEPFACE_TO_SHARED.get(face_top, face_top)

        # --- Fusion ---
        fused_pred = None
        fusion_result = fusion.fuse(face_result, audio_result)
        fused_pred = fusion_result.emotion

        results.append({
            'filename': fname,
            'ground_truth': ground_truth,
            'face_pred': face_pred,
            'audio_pred': audio_pred,
            'fused_pred': fused_pred,
            'face_conf': face_result['confidence'] if face_result else None,
            'audio_conf': audio_result.get('confidence') if audio_result else None,
            'fused_conf': fusion_result.confidence,
            'face_weight': fusion_result.face_weight,
            'audio_weight': fusion_result.audio_weight,
        })

    total_time = time.time() - start_time
    print(f"\n\nProcessed {len(results)} videos in {total_time:.1f}s "
          f"({total_time/len(results):.1f}s per video)")

    # ========================================================================
    # Results
    # ========================================================================

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Per-emotion accuracy
    emotions = sorted(set(r['ground_truth'] for r in results))

    print(f"\n{'Emotion':<12} {'Count':>6} {'Face':>8} {'Audio':>8} {'Fused':>8}")
    print("-" * 50)

    face_correct = 0
    audio_correct = 0
    fused_correct = 0
    total = 0

    for em in emotions:
        em_results = [r for r in results if r['ground_truth'] == em]
        n = len(em_results)
        fc = sum(1 for r in em_results if r['face_pred'] == em)
        ac = sum(1 for r in em_results if r['audio_pred'] == em)
        uc = sum(1 for r in em_results if r['fused_pred'] == em)

        face_correct += fc
        audio_correct += ac
        fused_correct += uc
        total += n

        print(f"{em:<12} {n:>6} {fc/n:>7.0%} {ac/n:>7.0%} {uc/n:>7.0%}")

    print("-" * 50)
    print(f"{'Overall':<12} {total:>6} "
          f"{face_correct/total:>7.0%} "
          f"{audio_correct/total:>7.0%} "
          f"{fused_correct/total:>7.0%}")

    # Save detailed results to CSV
    csv_path = os.path.join(os.path.dirname(__file__), 'fusion_ravdess_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'filename', 'ground_truth', 'face_pred', 'audio_pred', 'fused_pred',
            'face_conf', 'audio_conf', 'fused_conf', 'face_weight', 'audio_weight'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed results saved to: {csv_path}")


if __name__ == "__main__":
    main()
