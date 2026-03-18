# Demos

GUI applications for testing calibration and fusion. All demos should be run from the `calibration_test/` root directory.

## Files

| File | Description | Command |
|------|-------------|---------|
| `visual_demo.py` | Face emotion calibration. Shows raw HSEmotion output vs calibrated output side-by-side. Calibrates 3 states (neutral, happy, calm) using webcam. | `venv/bin/python demos/visual_demo.py --camera 1` |
| `audio_demo.py` | Audio emotion calibration. Shows raw Emotion2Vec output vs calibrated output side-by-side. Uses emotion2vec_plus_large model with prediction smoothing. | `venv/bin/python demos/audio_demo.py` |
| `fusion_demo.py` | Multimodal fusion. Runs webcam + mic simultaneously. 5-column display: Face Raw, Face Cal, Audio Raw, Audio Cal, Fused. Toggle between V1 (probability fusion) and V2 (V-A space fusion). | `venv/bin/python demos/fusion_demo.py --camera 1` |
| `comparison_demo.py` | Compares three calibration approaches side-by-side on same camera feed: Embeddings (1280-dim, HSEmotion), Landmarks (1434-dim, MediaPipe), Action Units (8-dim, MediaPipe). Shows cosine similarity scores for each. | `venv/bin/python demos/comparison_demo.py --camera 1` |
| `actionunits_demo.py` | Sourick's standalone AU-based calibration using MediaPipe Face Mesh. OpenCV GUI (not Tkinter). Click to capture baselines. | `venv/bin/python demos/actionunits_demo.py` |

## Camera Selection

Most demos accept `--camera N`:
- `--camera 0` — Default camera (may be iPhone via Continuity Camera)
- `--camera 1` — Usually MacBook webcam
- `--camera 2` — OBS Virtual Camera or other

Check available cameras:
```bash
venv/bin/python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, _ = cap.read()
        print(f'Camera {i}: {\"OK\" if ret else \"No frame\"} ')
        cap.release()
"
```

## Calibration Flow (Same for All Demos)

1. Click **Calibrate** → Enter user ID
2. Show **Neutral** face/voice (5-10 sec)
3. Show **Happy** face/voice (5-10 sec)
4. Show **Calm** face/voice (5-10 sec)
5. System averages 20-25 frames per state, stores as user profile
6. Live detection compares embeddings to baselines via cosine similarity
