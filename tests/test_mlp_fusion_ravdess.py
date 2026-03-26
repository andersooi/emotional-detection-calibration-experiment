"""
Evaluate MLP fusion vs V3 rule-based fusion on cached features.

Compares: Face-only, Audio-only, V3 (rule-based), MLP (learned)
on both RAVDESS test set and elderly AI-generated clips.

Usage:
    PYTHONPATH=. venv/bin/python tests/test_mlp_fusion_ravdess.py

Requires:
    data/ravdess_test_features.npz
    data/elderly_features.npz (optional)
    models/mlp_fusion.pt
"""

import os
import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fusion import (
    SHARED_EMOTIONS, ProbabilityFusion, compute_modality_weights,
    align_face_probs, align_audio_probs,
)
from core.mlp_fusion import MLPFusion


# ============================================================================
# Configuration
# ============================================================================

TEST_FEATURES = "data/ravdess_test_features.npz"
ELDERLY_FEATURES = "data/elderly_features.npz"
MODEL_PATH = "models/mlp_fusion.pt"

# DeepFace labels for rebuilding face_result dicts
DEEPFACE_LABELS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_dataset(name: str, data: dict, mlp_fusion: MLPFusion):
    """Evaluate all approaches on one dataset."""
    face_vecs = data['face_vecs']
    audio_vecs = data['audio_vecs']
    labels = data['labels']
    label_names = data['label_names']
    shared_emotions = list(data['shared_emotions'])

    n = len(labels)
    v3_fusion = ProbabilityFusion()

    # Track predictions
    face_preds = []
    audio_preds = []
    v3_preds = []
    mlp_preds = []

    for i in range(n):
        face_vec = face_vecs[i]
        audio_vec = audio_vecs[i]
        gt_idx = labels[i]

        # Face-only prediction (argmax of aligned face probs)
        face_top_idx = int(np.argmax(face_vec))
        face_preds.append(shared_emotions[face_top_idx])

        # Audio-only prediction
        audio_top_idx = int(np.argmax(audio_vec))
        audio_preds.append(shared_emotions[audio_top_idx])

        # Build result dicts for fusion
        face_probs = {em: float(face_vec[j]) for j, em in enumerate(shared_emotions)}
        audio_probs = {em: float(audio_vec[j]) for j, em in enumerate(shared_emotions)}

        face_result = {
            'top_emotion': shared_emotions[face_top_idx],
            'confidence': float(face_vec[face_top_idx]),
            'emotion_probs': face_probs,
        }
        audio_result = {
            'top_emotion': shared_emotions[audio_top_idx],
            'confidence': float(audio_vec[audio_top_idx]),
            'emotion_probs': audio_probs,
        }

        # V3 fusion
        v3_result = v3_fusion.fuse(face_result, audio_result)
        v3_preds.append(v3_result.emotion)

        # MLP fusion
        mlp_result = mlp_fusion.fuse(face_result, audio_result)
        mlp_preds.append(mlp_result.emotion)

    # Compute accuracies
    gt_names = [str(ln) for ln in label_names]

    print(f"\n{'='*70}")
    print(f"Results: {name} ({n} samples)")
    print(f"{'='*70}")

    print(f"\n{'Emotion':<12} {'Count':>6} {'Face':>8} {'Audio':>8} {'V3':>8} {'MLP':>8}")
    print("-" * 55)

    emotions = sorted(set(gt_names))
    totals = {'face': 0, 'audio': 0, 'v3': 0, 'mlp': 0, 'n': 0}

    for em in emotions:
        idx = [i for i in range(n) if gt_names[i] == em]
        count = len(idx)
        fc = sum(1 for i in idx if face_preds[i] == em)
        ac = sum(1 for i in idx if audio_preds[i] == em)
        vc = sum(1 for i in idx if v3_preds[i] == em)
        mc = sum(1 for i in idx if mlp_preds[i] == em)

        totals['face'] += fc
        totals['audio'] += ac
        totals['v3'] += vc
        totals['mlp'] += mc
        totals['n'] += count

        print(f"{em:<12} {count:>6} {fc/count:>7.0%} {ac/count:>7.0%} "
              f"{vc/count:>7.0%} {mc/count:>7.0%}")

    print("-" * 55)
    tn = totals['n']
    print(f"{'Overall':<12} {tn:>6} "
          f"{totals['face']/tn:>7.0%} {totals['audio']/tn:>7.0%} "
          f"{totals['v3']/tn:>7.0%} {totals['mlp']/tn:>7.0%}")

    # Key metrics
    print(f"\nKey metrics:")
    sad_idx = [i for i in range(n) if gt_names[i] == 'Sad']
    if sad_idx:
        sad_face_wrong = [i for i in sad_idx if face_preds[i] != 'Sad']
        v3_sad_rescued = sum(1 for i in sad_face_wrong if v3_preds[i] == 'Sad')
        mlp_sad_rescued = sum(1 for i in sad_face_wrong if mlp_preds[i] == 'Sad')
        print(f"  Sad rescue (face wrong → fused correct): V3={v3_sad_rescued}/{len(sad_face_wrong)}, "
              f"MLP={mlp_sad_rescued}/{len(sad_face_wrong)}")

    happy_idx = [i for i in range(n) if gt_names[i] == 'Happy']
    if happy_idx:
        happy_face_right = [i for i in happy_idx if face_preds[i] == 'Happy']
        v3_happy_broken = sum(1 for i in happy_face_right if v3_preds[i] != 'Happy')
        mlp_happy_broken = sum(1 for i in happy_face_right if mlp_preds[i] != 'Happy')
        print(f"  Happy preserved (face right → fused right): V3 broken={v3_happy_broken}/{len(happy_face_right)}, "
              f"MLP broken={mlp_happy_broken}/{len(happy_face_right)}")


def main():
    # Load MLP model
    mlp = MLPFusion(model_path=MODEL_PATH)
    if not mlp.loaded:
        print(f"Warning: MLP model not found at {MODEL_PATH}. Using fallback (average).")

    # RAVDESS test set
    if os.path.exists(TEST_FEATURES):
        data = dict(np.load(TEST_FEATURES, allow_pickle=True))
        evaluate_dataset("RAVDESS Test Set", data, mlp)
    else:
        print(f"Test features not found: {TEST_FEATURES}")

    # Elderly clips
    if os.path.exists(ELDERLY_FEATURES):
        data = dict(np.load(ELDERLY_FEATURES, allow_pickle=True))
        evaluate_dataset("Elderly AI-Generated", data, mlp)
    else:
        print(f"\nElderly features not found: {ELDERLY_FEATURES}")
        print("(Run extract_features.py to generate)")


if __name__ == "__main__":
    main()
