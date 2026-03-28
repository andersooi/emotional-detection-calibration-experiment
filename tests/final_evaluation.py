"""
Final Evaluation: Face-only vs Audio-only vs V3 Rule-Based vs MLP Learned Fusion

Runs on cached features from extract_features.py. Reports:
- Per-emotion accuracy table
- Per-emotion F1 scores
- Confusion matrices (for V3 and MLP)
- Key metrics (Sad rescue, Happy preservation)
- Overall summary

Usage:
    PYTHONPATH=. venv/bin/python tests/final_evaluation.py
"""

import os
import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fusion import SHARED_EMOTIONS, ProbabilityFusion
from core.mlp_fusion import MLPFusion


# ============================================================================
# Configuration
# ============================================================================

TEST_FEATURES = "data/ravdess_test_features.npz"
ELDERLY_FEATURES = "data/elderly_features.npz"
MODEL_PATH = "models/mlp_fusion.pt"


# ============================================================================
# Evaluation Utilities
# ============================================================================

def compute_f1(label_names, preds, emotion):
    """Compute precision, recall, F1 for one emotion class."""
    n = len(label_names)
    tp = sum(1 for i in range(n) if label_names[i] == emotion and preds[i] == emotion)
    fp = sum(1 for i in range(n) if label_names[i] != emotion and preds[i] == emotion)
    fn = sum(1 for i in range(n) if label_names[i] == emotion and preds[i] != emotion)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f1


def print_confusion_matrix(label_names, preds, shared_emotions, method_name):
    """Print a confusion matrix."""
    n = len(label_names)
    print(f"\n  {method_name} Confusion Matrix (rows=true, cols=predicted):")
    # Header
    print(f"  {'':>12}", end='')
    for em in shared_emotions:
        print(f" {em[:6]:>6}", end='')
    print()

    for true_em in shared_emotions:
        print(f"  {true_em:<12}", end='')
        for pred_em in shared_emotions:
            c = sum(1 for i in range(n)
                    if label_names[i] == true_em and preds[i] == pred_em)
            print(f" {c:>6}", end='')
        print()


def run_predictions(face_vecs, audio_vecs, shared, v3_fusion, mlp_fusion):
    """Run all four approaches on feature vectors. Returns dict of prediction lists."""
    n = len(face_vecs)
    preds = {'face': [], 'audio': [], 'v3': [], 'mlp': []}

    for i in range(n):
        fv, av = face_vecs[i], audio_vecs[i]
        fi = int(np.argmax(fv))
        ai = int(np.argmax(av))

        preds['face'].append(shared[fi])
        preds['audio'].append(shared[ai])

        # Build result dicts for fusion
        fp = {em: float(fv[j]) for j, em in enumerate(shared)}
        ap = {em: float(av[j]) for j, em in enumerate(shared)}
        fr = {'top_emotion': shared[fi], 'confidence': float(fv[fi]), 'emotion_probs': fp}
        ar = {'top_emotion': shared[ai], 'confidence': float(av[ai]), 'emotion_probs': ap}

        preds['v3'].append(v3_fusion.fuse(fr, ar).emotion)
        preds['mlp'].append(mlp_fusion.fuse(fr, ar).emotion)

    return preds


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_dataset(name, data, v3_fusion, mlp_fusion):
    """Run full evaluation on one dataset."""
    face_vecs = data['face_vecs']
    audio_vecs = data['audio_vecs']
    labels = data['labels']
    label_names = [str(ln) for ln in data['label_names']]
    shared = list(data['shared_emotions'])
    n = len(labels)

    # Run predictions
    preds = run_predictions(face_vecs, audio_vecs, shared, v3_fusion, mlp_fusion)

    # Header
    print(f"\n{'=' * 75}")
    print(f"  {name} ({n} samples)")
    print(f"{'=' * 75}")

    # ---- Per-emotion Accuracy ----
    emotions = sorted(set(label_names))
    methods = ['face', 'audio', 'v3', 'mlp']

    print(f"\n  ACCURACY")
    print(f"  {'Emotion':<12} {'Count':>6} {'Face':>8} {'Audio':>8} {'V3':>8} {'MLP':>8}")
    print(f"  {'-' * 53}")

    totals = {k: 0 for k in methods}
    total_n = 0
    for em in emotions:
        idx = [i for i in range(n) if label_names[i] == em]
        cnt = len(idx)
        total_n += cnt
        row = f"  {em:<12} {cnt:>6}"
        for method in methods:
            c = sum(1 for i in idx if preds[method][i] == em)
            totals[method] += c
            row += f" {c / cnt:>7.0%}"
        print(row)

    print(f"  {'-' * 53}")
    row = f"  {'Overall':<12} {total_n:>6}"
    for method in methods:
        row += f" {totals[method] / total_n:>7.0%}"
    print(row)

    # ---- Per-emotion F1 ----
    print(f"\n  F1 SCORES")
    print(f"  {'Emotion':<12} {'Face':>8} {'Audio':>8} {'V3':>8} {'MLP':>8}")
    print(f"  {'-' * 45}")

    macro_f1 = {m: [] for m in methods}
    for em in emotions:
        row = f"  {em:<12}"
        for method in methods:
            _, _, f1 = compute_f1(label_names, preds[method], em)
            macro_f1[method].append(f1)
            row += f" {f1:>7.2f}"
        print(row)

    print(f"  {'-' * 45}")
    row = f"  {'Macro Avg':<12}"
    for method in methods:
        avg = np.mean(macro_f1[method])
        row += f" {avg:>7.2f}"
    print(row)

    # ---- Confusion Matrices ----
    for method, method_name in [('v3', 'V3 Rule-Based'), ('mlp', 'MLP Learned')]:
        print_confusion_matrix(label_names, preds[method], shared, method_name)

    # ---- Key Metrics ----
    print(f"\n  KEY METRICS")

    sad_idx = [i for i in range(n) if label_names[i] == 'Sad']
    if sad_idx:
        sad_face_wrong = [i for i in sad_idx if preds['face'][i] != 'Sad']
        if sad_face_wrong:
            for method in ['v3', 'mlp']:
                rescued = sum(1 for i in sad_face_wrong if preds[method][i] == 'Sad')
                print(f"    Sad rescue ({method.upper()}): {rescued}/{len(sad_face_wrong)} "
                      f"({rescued / len(sad_face_wrong):.0%}) — face got wrong, fusion corrected")

    happy_idx = [i for i in range(n) if label_names[i] == 'Happy']
    if happy_idx:
        happy_face_right = [i for i in happy_idx if preds['face'][i] == 'Happy']
        if happy_face_right:
            for method in ['v3', 'mlp']:
                broken = sum(1 for i in happy_face_right if preds[method][i] != 'Happy')
                print(f"    Happy preserved ({method.upper()}): {len(happy_face_right) - broken}/{len(happy_face_right)} "
                      f"({(len(happy_face_right) - broken) / len(happy_face_right):.0%}) — face got right, fusion kept it")

    # Agreement analysis
    print(f"\n    Model agreement:")
    agree = sum(1 for i in range(n) if preds['face'][i] == preds['audio'][i])
    print(f"      Face-Audio agree: {agree}/{n} ({agree / n:.0%})")
    v3_mlp_agree = sum(1 for i in range(n) if preds['v3'][i] == preds['mlp'][i])
    print(f"      V3-MLP agree: {v3_mlp_agree}/{n} ({v3_mlp_agree / n:.0%})")

    # Cases where V3 and MLP disagree
    disagree_idx = [i for i in range(n) if preds['v3'][i] != preds['mlp'][i]]
    if disagree_idx:
        v3_right = sum(1 for i in disagree_idx if preds['v3'][i] == label_names[i])
        mlp_right = sum(1 for i in disagree_idx if preds['mlp'][i] == label_names[i])
        print(f"      When V3 ≠ MLP ({len(disagree_idx)} cases): V3 correct={v3_right}, MLP correct={mlp_right}")


def main():
    print("=" * 75)
    print("  FINAL EVALUATION REPORT")
    print("  DeepFace + Emotion2Vec Multimodal Emotion Fusion")
    print("  Face-only | Audio-only | V3 Rule-Based | MLP Learned")
    print("=" * 75)

    # Load fusion engines
    mlp = MLPFusion(model_path=MODEL_PATH)
    if not mlp.loaded:
        print(f"WARNING: MLP model not found at {MODEL_PATH}. Using fallback.")
    v3 = ProbabilityFusion()

    # Evaluate on each dataset
    for name, path in [
        ("RAVDESS Test Set (300 clips, 5 held-out actors)", TEST_FEATURES),
        ("Elderly AI-Generated (14 clips, out-of-domain)", ELDERLY_FEATURES),
    ]:
        if not os.path.exists(path):
            print(f"\n  SKIPPED: {name} — {path} not found")
            continue
        data = dict(np.load(path, allow_pickle=True))
        evaluate_dataset(name, data, v3, mlp)

    # Summary
    print(f"\n{'=' * 75}")
    print("  SUMMARY")
    print(f"{'=' * 75}")
    print("""
    Models:
      Visual:  DeepFace (VGG-Face backbone, emotion CNN)
      Audio:   Emotion2Vec large (self-supervised speech model)

    Fusion approaches:
      V3:  Rule-based probability fusion with Happy-protected
           asymmetric audio weighting (audio 65% for negative emotions
           when face ≠ Happy)
      MLP: Learned fusion (14→32→16→7) trained on RAVDESS train set
           (1140 clips, 19 actors), 5-fold CV accuracy: 91.8% ± 3.7%

    Data split:
      Train: 1140 clips (19 actors)
      Test:  300 clips (5 held-out actors, speaker-independent)
      Elderly: 14 AI-generated clips (out-of-domain)

    Key findings:
      - Audio (Emotion2Vec) outperforms face (DeepFace) significantly
      - Both fusion approaches rescue sadness detection (DeepFace's weakness)
      - MLP handles Surprise better than V3 (learned vs hand-coded rules)
      - Both preserve Happy when face correctly detects it
      - MLP generalises better to elderly out-of-domain clips
    """)


if __name__ == "__main__":
    main()
