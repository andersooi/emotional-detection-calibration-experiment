# Core

Core calibration and fusion logic. No GUI — these are imported by the demo applications.

## Files

| File | Description |
|------|-------------|
| `calibration_visual.py` | HSEmotion facial emotion extractor (1280-dim embeddings, 8 emotion probs, valence-arousal), `UserBaseline` dataclass, `CalibrationManager` for profile persistence, `CalibratedDetector` with hybrid calibration logic (cosine similarity + raw override for non-calibrated emotions) |
| `calibration_audio.py` | Emotion2Vec speech emotion extractor (768-dim embeddings, 9 emotion probs), `AudioUserBaseline` dataclass, `AudioCalibrationManager`, `CalibratedAudioDetector` with the same hybrid calibration approach |
| `calibration_base.py` | Model-agnostic abstract interfaces (`BaseEmotionExtractor`, `GenericBaseline`, `GenericCalibrationManager`, `GenericCalibratedDetector`). Allows any new model to plug into the calibration pipeline by implementing `load()` and `extract()`. Includes adapter wrappers for HSEmotion and Emotion2Vec. |
| `fusion.py` | Late fusion strategies for combining visual and audio predictions. V1 (`ProbabilityFusion`): confidence-weighted average of aligned emotion probabilities. V2 (`VAFusion`): confidence-weighted fusion in valence-arousal space using Russell's Circumplex coordinates. `MultimodalFusion` wrapper for runtime version toggling. Signal-based modality weighting (downweights audio when it outputs Neutral). |
| `__init__.py` | Re-exports all public classes and functions for convenient imports: `from core import HSEmotionExtractor, MultimodalFusion, ...` |

## Usage

```python
from core import (
    HSEmotionExtractor, CalibratedDetector, UserBaseline,
    Emotion2VecExtractor, CalibratedAudioDetector,
    MultimodalFusion, FusionResult,
    BaseEmotionExtractor, GenericBaseline, GenericCalibratedDetector,
)
```

## Key Thresholds

| Threshold | Value | Location | Purpose |
|-----------|-------|----------|---------|
| Happy/Calm similarity | 0.80 | calibration_visual.py, calibration_audio.py | Cosine similarity to trigger calibrated prediction |
| Neutral similarity | 0.85 | calibration_visual.py, calibration_audio.py | Stricter threshold for neutral state |
| Raw override confidence | 0.60 | calibration_visual.py, calibration_audio.py | Trust raw model for non-calibrated emotions above this |
| V-A quadrant dead-zone | 0.10 | fusion.py | V-A values within +/-0.10 classified as Neutral |
| Audio Neutral weight | 0.15 | fusion.py | Weight given to audio when it outputs Neutral |
