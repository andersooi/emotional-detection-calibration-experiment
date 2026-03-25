# Emotion Detection with Calibration Experiment

Real-time multimodal emotion detection prototype for the AIRA elderly care robot. Combines facial and speech emotion models with per-user calibration and late fusion.

## Architecture

```
Face (webcam) ──> DeepFace (calibrated neutral + happy)
                         │
                    Fusion Adapter ──> ProbabilityFusion ──> Fused Emotion
                         │                                   (Q1-Q4 + Neutral)
Audio (mic) ───> Emotion2Vec large (raw) ──────────────┘
                         │
                    FunASR VAD (speech detection)
```

- **Visual model**: DeepFace (VGG-Face backbone) — calibrated for neutral/happy via cosine similarity on 1024-dim emotion embeddings
- **Audio model**: Emotion2Vec large — raw output, no calibration. Strong on sadness (80% on RAVDESS)
- **Fusion**: V1 Probability Fusion with asymmetric sadness weighting (audio gets 65% weight when it detects Sad but face says Neutral/Happy)
- **VAD**: FunASR FSMN VAD filters background noise so only speech reaches Emotion2Vec

## Quick Start

```bash
cd calibration_test

# Main fusion demo (DeepFace + Emotion2Vec)
PYTHONPATH=. venv/bin/python demos/deepface_audio_fusion_demo.py --camera 1

# HSEmotion visual-only demo
venv/bin/python demos/visual_demo.py --camera 1

# DeepFace embedding calibration demo
PYTHONPATH=. venv/bin/python demos/deepface_emb_demo.py --camera 1
```

## Project Structure

```
calibration_test/
├── core/                    # Core calibration, fusion, and adapter logic
│   ├── calibration_visual.py    # HSEmotion extractor + calibration
│   ├── calibration_audio.py     # Emotion2Vec extractor + calibration
│   ├── calibration_base.py      # Model-agnostic interfaces + DeepFace extractors
│   ├── fusion.py                # Late fusion (V1 probability, V2 V-A space)
│   ├── deepface_fusion_adapter.py  # Converts calibrated predictions for fusion
│   └── __init__.py              # Re-exports
├── demos/                   # GUI demo applications
│   ├── deepface_audio_fusion_demo.py  # Main: DeepFace + Emotion2Vec fusion
│   ├── visual_demo.py          # HSEmotion calibration
│   ├── deepface_emb_demo.py    # DeepFace embedding calibration
│   ├── deepface_logit_demo.py  # DeepFace neutral/smile score experiment
│   ├── deepface_demo.py        # DeepFace identity embedding (neutral-only)
│   ├── audio_demo.py           # Emotion2Vec audio calibration
│   ├── fusion_demo.py          # HSEmotion + Emotion2Vec fusion (reference)
│   └── comparison_demo.py      # Embedding vs landmark vs AU comparison
├── tests/                   # Test scripts
│   └── test_vad_live.py        # Standalone VAD test on live mic
├── user_profiles/           # Saved calibration profiles (.pkl)
└── venv/                    # Python virtual environment
```

## Calibration Flow

1. Click **Calibrate Face** → Enter user ID
2. Show **Neutral** face (5 sec) — anchors resting face
3. Dialog confirms → Click OK
4. Show **Happy** face (5 sec) — establishes positive boundary
5. System computes adaptive thresholds from inter-baseline similarity
6. Live detection: calibration for neutral/happy, raw model for other emotions

## Key Findings

- **DeepFace**: Good raw classifier. Calibration works for neutral + happy. Embeddings don't separate negative emotions well.
- **HSEmotion**: Better embeddings for calibration (1280-dim, emotion-trained). Native V-A output. Raw output already quite good.
- **Emotion2Vec large**: Strong on sadness (80% RAVDESS). Defaults to Neutral on natural speech — handled by signal-based weighting.
- **Fusion**: Asymmetric weighting lets audio carry the sadness signal that DeepFace misses.

## Camera Selection

Most demos accept `--camera N`:
- `--camera 0` — Default camera
- `--camera 1` — Usually MacBook webcam
