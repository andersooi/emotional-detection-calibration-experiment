"""
Late fusion strategies for multimodal emotion recognition.

Version 1: Probability Fusion - confidence-weighted average of aligned emotion probabilities.
Version 2: Confidence-Weighted V-A Fusion - fuse valence-arousal coordinates from both modalities.

Imports from:
    calibration_core: HSEmotionExtractor, CalibratedDetector
    calibration_core_audio: Emotion2VecExtractor, CalibratedAudioDetector
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# ============================================================================
# Constants
# ============================================================================

# 7 shared emotion categories between HSEmotion (8) and Emotion2Vec (9)
SHARED_EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Face model label → shared label
# Supports HSEmotion (8 classes) and DeepFace (7 classes).
# Also tolerates calibrated labels (Happy, Sad) that may differ from raw (Happiness, Sadness).
FACE_TO_SHARED = {
    'Anger': 'Angry',
    'Angry': 'Angry',
    'Disgust': 'Disgust',
    'Fear': 'Fear',
    'Happiness': 'Happy',
    'Happy': 'Happy',
    'Neutral': 'Neutral',
    'Sadness': 'Sad',
    'Sad': 'Sad',
    'Surprise': 'Surprise',
}

# Emotion2Vec label → shared label (drops Other, Unknown)
AUDIO_TO_SHARED = {
    'Angry': 'Angry',
    'Disgust': 'Disgust',
    'Fear': 'Fear',
    'Happy': 'Happy',
    'Neutral': 'Neutral',
    'Sad': 'Sad',
    'Surprise': 'Surprise',
}

# Standard V-A coordinates for emotion categories (Russell's Circumplex)
# Used to map Emotion2Vec discrete predictions into V-A space
EMOTION_VA_LOOKUP = {
    'Happy':    ( 0.8,  0.5),   # Q1: positive, activated
    'Angry':    (-0.6,  0.7),   # Q2: negative, activated
    'Fear':     (-0.6,  0.6),   # Q2: negative, activated
    'Disgust':  (-0.6,  0.3),   # Q2: negative, mild activation
    'Surprise': ( 0.2,  0.7),   # Q1: mild positive, highly activated
    'Sad':      (-0.7, -0.3),   # Q3: negative, deactivated
    'Neutral':  ( 0.0,  0.0),   # Center
    'Other':    ( 0.0,  0.0),   # Treat as neutral
    'Unknown':  ( 0.0,  0.0),   # Treat as neutral
}

QUADRANT_LABELS = {
    'Q1': 'Q1 (Happy/Excited)',
    'Q2': 'Q2 (Angry/Anxious)',
    'Q3': 'Q3 (Sad/Depressed)',
    'Q4': 'Q4 (Calm/Content)',
    'Neutral': 'Neutral'
}


# ============================================================================
# Utility Functions
# ============================================================================

def va_to_quadrant(valence: float, arousal: float, threshold: float = 0.1) -> str:
    """Map V-A coordinates to Russell's circumplex quadrant."""
    if abs(valence) < threshold and abs(arousal) < threshold:
        return 'Neutral'
    elif valence >= 0 and arousal >= 0:
        return 'Q1'
    elif valence < 0 and arousal >= 0:
        return 'Q2'
    elif valence < 0 and arousal < 0:
        return 'Q3'
    else:  # valence >= 0 and arousal < 0
        return 'Q4'


def align_face_probs(face_probs: Dict[str, float]) -> Dict[str, float]:
    """
    Align HSEmotion 8-class probabilities to 7 shared categories.
    Drops Contempt and renormalizes.

    Sums probabilities when multiple face labels alias to the same shared
    class (e.g. Anger + Angry → Angry) so DeepFace / HSEmotion keys are not
    overwritten by a later alias with mass 0.
    """
    aligned = {em: 0.0 for em in SHARED_EMOTIONS}
    for face_name, shared_name in FACE_TO_SHARED.items():
        aligned[shared_name] += face_probs.get(face_name, 0.0)

    # Renormalize
    total = sum(aligned.values())
    if total > 0:
        aligned = {k: v / total for k, v in aligned.items()}

    return aligned


def align_audio_probs(audio_probs: Dict[str, float]) -> Dict[str, float]:
    """
    Align Emotion2Vec 9-class probabilities to 7 shared categories.
    Drops Other and Unknown and renormalizes.
    """
    aligned = {}
    for audio_name, shared_name in AUDIO_TO_SHARED.items():
        aligned[shared_name] = audio_probs.get(audio_name, 0.0)

    # Renormalize
    total = sum(aligned.values())
    if total > 0:
        aligned = {k: v / total for k, v in aligned.items()}

    return aligned


def audio_probs_to_va(audio_probs: Dict[str, float]) -> Tuple[float, float]:
    """
    Convert Emotion2Vec probabilities to V-A coordinates.
    Uses probability-weighted sum over EMOTION_VA_LOOKUP.
    """
    valence = 0.0
    arousal = 0.0

    for emotion, prob in audio_probs.items():
        if emotion in EMOTION_VA_LOOKUP:
            v, a = EMOTION_VA_LOOKUP[emotion]
            valence += prob * v
            arousal += prob * a

    return valence, arousal


# ============================================================================
# FusionResult
# ============================================================================

@dataclass
class FusionResult:
    """Container for fusion output."""
    emotion: str
    confidence: float
    quadrant: str
    quadrant_label: str

    # V-A coordinates (V2 only; None for V1)
    fused_valence: Optional[float]
    fused_arousal: Optional[float]

    # Per-modality info
    face_emotion: Optional[str]
    face_confidence: Optional[float]
    audio_emotion: Optional[str]
    audio_confidence: Optional[float]

    # Fusion metadata
    fusion_version: int
    face_weight: float
    audio_weight: float
    modalities_present: str  # 'both', 'face_only', 'audio_only', 'none'


def compute_modality_weights(
    face_emotion: Optional[str], face_conf: float,
    audio_emotion: Optional[str], audio_conf: float
) -> tuple:
    """
    Signal-based modality weighting with asymmetric sadness bias.

    Key insights:
    - When audio says Neutral, it's usually "I don't know" → downweight.
    - When audio detects sadness but face says neutral/happy, audio is likely
      catching something DeepFace misses → give audio more weight.
    - When both agree, equal trust.
    """
    if audio_emotion is None or audio_emotion == 'Neutral':
        # Audio has no signal — lean heavily on face
        return 0.85, 0.15

    face_shared = FACE_TO_SHARED.get(face_emotion, face_emotion) if face_emotion else None

    if face_shared is not None and audio_emotion == face_shared:
        # Both agree on the same emotion — equal trust
        return 0.50, 0.50

    # Audio detects negative emotion with high confidence → trust audio more.
    # DeepFace struggles with negative emotions (0-18% on Disgust/Fear/Sad)
    # and often misclassifies them as other emotions (e.g., Disgust→Angry).
    # Emotion2Vec is consistently stronger (~50-88%) on these categories.
    if (audio_emotion in ('Sad', 'Angry', 'Fear', 'Disgust') and audio_conf >= 0.50):
        return 0.35, 0.65

    # General disagreement — audio has a non-neutral signal
    return 0.55, 0.45


def _default_result(fusion_version: int) -> FusionResult:
    """Return a default Neutral result when no modalities are present."""
    return FusionResult(
        emotion='Neutral', confidence=0.0,
        quadrant='Neutral', quadrant_label=QUADRANT_LABELS['Neutral'],
        fused_valence=None, fused_arousal=None,
        face_emotion=None, face_confidence=None,
        audio_emotion=None, audio_confidence=None,
        fusion_version=fusion_version,
        face_weight=0.0, audio_weight=0.0,
        modalities_present='none'
    )


# ============================================================================
# Version 1: Probability Fusion
# ============================================================================

class ProbabilityFusion:
    """Confidence-weighted average of aligned emotion probabilities."""

    def fuse(self, face_result: Optional[Dict], audio_result: Optional[Dict]) -> FusionResult:
        # Neither present
        if face_result is None and audio_result is None:
            return _default_result(fusion_version=1)

        # Face only
        if face_result is not None and audio_result is None:
            aligned = align_face_probs(face_result['emotion_probs'])
            top = max(aligned, key=aligned.get)
            q = va_to_quadrant(*EMOTION_VA_LOOKUP.get(top, (0.0, 0.0)))
            return FusionResult(
                emotion=top, confidence=face_result['confidence'],
                quadrant=q, quadrant_label=QUADRANT_LABELS[q],
                fused_valence=None, fused_arousal=None,
                face_emotion=face_result['top_emotion'], face_confidence=face_result['confidence'],
                audio_emotion=None, audio_confidence=None,
                fusion_version=1, face_weight=1.0, audio_weight=0.0,
                modalities_present='face_only'
            )

        # Audio only
        if face_result is None and audio_result is not None:
            aligned = align_audio_probs(audio_result['emotion_probs'])
            top = max(aligned, key=aligned.get)
            q = va_to_quadrant(*EMOTION_VA_LOOKUP.get(top, (0.0, 0.0)))
            return FusionResult(
                emotion=top, confidence=audio_result['confidence'],
                quadrant=q, quadrant_label=QUADRANT_LABELS[q],
                fused_valence=None, fused_arousal=None,
                face_emotion=None, face_confidence=None,
                audio_emotion=audio_result['top_emotion'], audio_confidence=audio_result['confidence'],
                fusion_version=1, face_weight=0.0, audio_weight=1.0,
                modalities_present='audio_only'
            )

        # Both present: signal-based weighted fusion
        face_conf = face_result['confidence']
        audio_conf = audio_result['confidence']

        fw, aw = compute_modality_weights(
            face_result['top_emotion'], face_conf,
            audio_result['top_emotion'], audio_conf
        )

        face_aligned = align_face_probs(face_result['emotion_probs'])
        audio_aligned = align_audio_probs(audio_result['emotion_probs'])

        fused_probs = {}
        for em in SHARED_EMOTIONS:
            fused_probs[em] = fw * face_aligned.get(em, 0.0) + aw * audio_aligned.get(em, 0.0)

        top = max(fused_probs, key=fused_probs.get)
        fused_conf = fused_probs[top]
        q = va_to_quadrant(*EMOTION_VA_LOOKUP.get(top, (0.0, 0.0)))

        return FusionResult(
            emotion=top, confidence=fused_conf,
            quadrant=q, quadrant_label=QUADRANT_LABELS[q],
            fused_valence=None, fused_arousal=None,
            face_emotion=face_result['top_emotion'], face_confidence=face_conf,
            audio_emotion=audio_result['top_emotion'], audio_confidence=audio_conf,
            fusion_version=1, face_weight=fw, audio_weight=aw,
            modalities_present='both'
        )


# ============================================================================
# Version 2: Confidence-Weighted V-A Fusion
# ============================================================================

class VAFusion:
    """Confidence-weighted fusion in Valence-Arousal space."""

    def __init__(self):
        self.face_neutral_valence: float = 0.0
        self.face_neutral_arousal: float = 0.0
        self.calibration_active: bool = False

    def set_face_calibration(self, neutral_valence: float, neutral_arousal: float):
        """Set face calibration baseline for V-A shifting."""
        self.face_neutral_valence = neutral_valence
        self.face_neutral_arousal = neutral_arousal
        self.calibration_active = True

    def clear_calibration(self):
        """Remove face calibration offset."""
        self.face_neutral_valence = 0.0
        self.face_neutral_arousal = 0.0
        self.calibration_active = False

    def _get_face_va(self, face_result: Dict) -> Tuple[float, float]:
        """Get face V-A, applying calibration shift if active."""
        v = face_result.get('valence', 0.0)
        a = face_result.get('arousal', 0.0)

        if self.calibration_active:
            v = v - self.face_neutral_valence
            a = a - self.face_neutral_arousal

        return v, a

    def _emotion_from_va(self, valence: float, arousal: float) -> str:
        """Derive emotion label from V-A coordinates by finding nearest in lookup."""
        best_emotion = 'Neutral'
        best_dist = float('inf')

        for emotion, (v, a) in EMOTION_VA_LOOKUP.items():
            if emotion in ('Other', 'Unknown'):
                continue
            dist = (valence - v) ** 2 + (arousal - a) ** 2
            if dist < best_dist:
                best_dist = dist
                best_emotion = emotion

        return best_emotion

    def fuse(self, face_result: Optional[Dict], audio_result: Optional[Dict]) -> FusionResult:
        # Neither present
        if face_result is None and audio_result is None:
            return _default_result(fusion_version=2)

        # Face only
        if face_result is not None and audio_result is None:
            fv, fa = self._get_face_va(face_result)
            q = va_to_quadrant(fv, fa)
            emotion = self._emotion_from_va(fv, fa)
            return FusionResult(
                emotion=emotion, confidence=face_result['confidence'],
                quadrant=q, quadrant_label=QUADRANT_LABELS[q],
                fused_valence=fv, fused_arousal=fa,
                face_emotion=face_result['top_emotion'], face_confidence=face_result['confidence'],
                audio_emotion=None, audio_confidence=None,
                fusion_version=2, face_weight=1.0, audio_weight=0.0,
                modalities_present='face_only'
            )

        # Audio only
        if face_result is None and audio_result is not None:
            av, aa = audio_probs_to_va(audio_result['emotion_probs'])
            q = va_to_quadrant(av, aa)
            emotion = self._emotion_from_va(av, aa)
            return FusionResult(
                emotion=emotion, confidence=audio_result['confidence'],
                quadrant=q, quadrant_label=QUADRANT_LABELS[q],
                fused_valence=av, fused_arousal=aa,
                face_emotion=None, face_confidence=None,
                audio_emotion=audio_result['top_emotion'], audio_confidence=audio_result['confidence'],
                fusion_version=2, face_weight=0.0, audio_weight=1.0,
                modalities_present='audio_only'
            )

        # Both present: signal-based weighted V-A fusion
        face_conf = face_result['confidence']
        audio_conf = audio_result['confidence']

        fw, aw = compute_modality_weights(
            face_result['top_emotion'], face_conf,
            audio_result['top_emotion'], audio_conf
        )

        fv, fa = self._get_face_va(face_result)
        av, aa = audio_probs_to_va(audio_result['emotion_probs'])

        fused_v = fw * fv + aw * av
        fused_a = fw * fa + aw * aa

        q = va_to_quadrant(fused_v, fused_a)
        emotion = self._emotion_from_va(fused_v, fused_a)

        return FusionResult(
            emotion=emotion, confidence=max(face_conf, audio_conf),
            quadrant=q, quadrant_label=QUADRANT_LABELS[q],
            fused_valence=fused_v, fused_arousal=fused_a,
            face_emotion=face_result['top_emotion'], face_confidence=face_conf,
            audio_emotion=audio_result['top_emotion'], audio_confidence=audio_conf,
            fusion_version=2, face_weight=fw, audio_weight=aw,
            modalities_present='both'
        )


# ============================================================================
# Unified Interface
# ============================================================================

class MultimodalFusion:
    """Unified interface for toggling between fusion versions."""

    def __init__(self):
        self.v1 = ProbabilityFusion()
        self.v2 = VAFusion()
        self.active_version: int = 1

    def set_version(self, version: int):
        """Switch between Version 1 (probability) and Version 2 (V-A)."""
        assert version in (1, 2), f"Version must be 1 or 2, got {version}"
        self.active_version = version

    def set_face_calibration(self, neutral_valence: float, neutral_arousal: float):
        """Pass calibration data to V2."""
        self.v2.set_face_calibration(neutral_valence, neutral_arousal)

    def clear_calibration(self):
        """Clear calibration from V2."""
        self.v2.clear_calibration()

    def fuse(self, face_result: Optional[Dict], audio_result: Optional[Dict]) -> FusionResult:
        """Run fusion using the currently active version."""
        if self.active_version == 1:
            return self.v1.fuse(face_result, audio_result)
        else:
            return self.v2.fuse(face_result, audio_result)
