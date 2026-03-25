# Core

Core calibration, fusion, and adapter logic. No GUI — these are imported by the demo applications.

## Files

| File | Description |
|------|-------------|
| `calibration_visual.py` | HSEmotion facial emotion extractor (1280-dim embeddings, 8 emotion probs, valence-arousal). `UserBaseline` dataclass, `CalibrationManager` for profile persistence, `CalibratedDetector` with hybrid calibration logic (cosine similarity + V-A shift sadness detection + raw override for non-calibrated emotions). Adaptive V-A thresholds from neutral capture variance. |
| `calibration_audio.py` | Emotion2Vec speech emotion extractor (768-dim embeddings, 9 emotion probs). `AudioUserBaseline` dataclass, `AudioCalibrationManager`, `CalibratedAudioDetector` with the same hybrid calibration approach. |
| `calibration_base.py` | Model-agnostic abstract interfaces (`BaseEmotionExtractor`, `GenericBaseline`, `GenericCalibrationManager`, `GenericCalibratedDetector`). Includes `DeepFaceEmotionEmbeddingExtractor` (1024-dim from penultimate Dense layer) and `DeepFaceExtractor` (probability vector). Utility functions: `cosine_similarity`, `compute_adaptive_thresholds`. |
| `fusion.py` | Late fusion strategies. V1 (`ProbabilityFusion`): confidence-weighted average of aligned emotion probabilities with asymmetric sadness weighting (audio 65% when Sad + face Neutral/Happy). V2 (`VAFusion`): confidence-weighted fusion in V-A space. Signal-based modality weighting. Tolerant label mapping supporting both HSEmotion and DeepFace label formats. |
| `deepface_fusion_adapter.py` | Converts DeepFace calibrated predictions into coherent probability vectors for fusion. When calibration fires, boosts the calibrated class probability and renormalizes so fusion sees a distribution consistent with the calibrated decision. |
| `__init__.py` | Re-exports all public classes and functions for convenient imports. |

## Usage

```python
from core import (
    # Visual models
    HSEmotionExtractor, CalibratedDetector, UserBaseline,
    DeepFaceEmotionEmbeddingExtractor, DeepFaceExtractor,
    # Audio model
    Emotion2VecExtractor, CalibratedAudioDetector,
    # Fusion
    ProbabilityFusion, MultimodalFusion, FusionResult,
    build_face_result,
    # Generic interfaces
    BaseEmotionExtractor, GenericBaseline, GenericCalibratedDetector,
    cosine_similarity, compute_adaptive_thresholds,
)
```

## Key Thresholds

| Threshold | Default | Location | Purpose |
|-----------|---------|----------|---------|
| Similarity threshold | Adaptive (sim_nh * 0.85) | calibration_visual.py, calibration_base.py | Cosine similarity to trigger calibrated prediction |
| Neutral threshold | Adaptive (sim_nh * 0.87) | calibration_visual.py, calibration_base.py | Stricter threshold for neutral state |
| Deviation floor | Adaptive (sim_nh * 0.78) | calibration_visual.py, calibration_base.py | Below this, user is in uncalibrated territory |
| Raw override confidence | 0.60 | calibration_visual.py, calibration_base.py | Trust raw model for non-calibrated emotions above this |
| V-A strong override | Adaptive (-3σ) | calibration_visual.py | V-A shift overrides calibration for obvious sadness |
| V-A moderate shift | Adaptive (-2σ) | calibration_visual.py | V-A shift fires after calibration miss |
| Audio Neutral weight | 0.15 | fusion.py | Weight given to audio when it outputs Neutral |
| Audio Sad weight | 0.65 | fusion.py | Weight given to audio when it detects Sad but face says Neutral/Happy |
