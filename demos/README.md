# Demos

GUI applications for testing calibration and fusion. All demos should be run from the `calibration_test/` root directory.

## Files

| File | Description | Command |
|------|-------------|---------|
| `deepface_audio_fusion_demo.py` | **Main demo.** DeepFace (calibrated) + Emotion2Vec (raw) fusion. FunASR VAD for speech detection. Face EMA smoothing. Shows face raw/adapted probs, audio probs, fusion weights, and fused output. | `PYTHONPATH=. venv/bin/python demos/deepface_audio_fusion_demo.py --camera 1` |
| `visual_demo.py` | HSEmotion face calibration. Raw vs calibrated side-by-side with V-A shift detection, adaptive thresholds, and probability bars. Calibrates neutral + happy. | `venv/bin/python demos/visual_demo.py --camera 1` |
| `deepface_emb_demo.py` | DeepFace 1024-dim emotion embedding calibration. Neutral + happy with 2-state adaptive thresholds. Probs from DeepFace full pipeline. | `PYTHONPATH=. venv/bin/python demos/deepface_emb_demo.py --camera 1` |
| `deepface_logit_demo.py` | DeepFace neutral/smile score experiment. Tests smile_score = log(p_happy) - log(p_neutral) as calibration signal. Finding: insufficient separation for subtle smiles. | `PYTHONPATH=. venv/bin/python demos/deepface_logit_demo.py --camera 1` |
| `deepface_demo.py` | DeepFace identity embedding (VGG-Face 4096-dim) neutral-only calibration with variance-based threshold. | `PYTHONPATH=. venv/bin/python demos/deepface_demo.py --camera 1` |
| `audio_demo.py` | Emotion2Vec audio calibration. Raw vs calibrated with prediction smoothing (majority vote + hysteresis). | `venv/bin/python demos/audio_demo.py` |
| `fusion_demo.py` | HSEmotion + Emotion2Vec fusion (reference). 5-column display. V1/V2 toggle. Kept as comparison baseline. | `venv/bin/python demos/fusion_demo.py --camera 1` |
| `comparison_demo.py` | Compares calibration approaches: Embeddings (HSEmotion 1280-dim) vs Landmarks (MediaPipe 1434-dim) vs Action Units (MediaPipe 8-dim). | `venv/bin/python demos/comparison_demo.py --camera 1` |
| `actionunits_demo.py` | Sourick's standalone AU-based calibration using MediaPipe Face Mesh. OpenCV GUI. | `venv/bin/python demos/actionunits_demo.py` |

## Camera Selection

Most demos accept `--camera N`:
- `--camera 0` — Default camera (may be iPhone via Continuity Camera)
- `--camera 1` — Usually MacBook webcam
- `--camera 2` — OBS Virtual Camera or other

## Calibration Flow

### Face calibration (visual demos + fusion demo)
1. Click **Calibrate** / **Calibrate Face** → Enter user ID
2. Show **Neutral** face (5 sec) — anchors resting face
3. Confirmation dialog → Click OK
4. Show **Happy** face (5 sec) — establishes positive boundary
5. System computes adaptive thresholds from inter-baseline similarity
6. Live: calibration for neutral/happy, raw model fallback for other emotions

### Audio (audio_demo.py only)
1. Click **Calibrate** → Enter user ID
2. Count 1-10 in **Neutral** voice (8 sec)
3. Talk about something **Happy** (10 sec)
4. Speak calmly for **Calm** (10 sec)
