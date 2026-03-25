"""
Core calibration and fusion logic.

Re-exports all public classes and functions for convenient imports:
    from core import HSEmotionExtractor, CalibratedDetector, ...
"""

from core.calibration_visual import (
    UserBaseline,
    HSEmotionExtractor,
    CalibrationManager,
    CalibratedDetector,
    average_embeddings,
    average_values,
)

from core.calibration_audio import (
    AudioUserBaseline,
    Emotion2VecExtractor,
    AudioCalibrationManager,
    CalibratedAudioDetector,
)

from core.fusion import (
    MultimodalFusion,
    ProbabilityFusion,
    VAFusion,
    FusionResult,
    SHARED_EMOTIONS,
    QUADRANT_LABELS,
    EMOTION_VA_LOOKUP,
    align_face_probs,
    align_audio_probs,
    audio_probs_to_va,
    va_to_quadrant,
    compute_modality_weights,
)

from core.calibration_base import (
    BaseEmotionExtractor,
    GenericBaseline,
    GenericCalibrationManager,
    GenericCalibratedDetector,
    HSEmotionExtractorAdapter,
    Emotion2VecExtractorAdapter,
    DeepFaceExtractor,
    DeepFaceEmotionEmbeddingExtractor,
    cosine_similarity,
    compute_adaptive_thresholds,
)

from core.deepface_fusion_adapter import build_face_result
