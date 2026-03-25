"""
Self-contained calibration logic for HSEmotion testing.

This module provides:
- UserBaseline: Dataclass storing calibration data
- HSEmotionExtractor: Extract embeddings and V-A from HSEmotion
- CalibrationManager: Save/load user profiles
- CalibratedDetector: Compare raw vs calibrated predictions
"""

import os
import pickle
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple, List


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class UserBaseline:
    """Stores calibration data for one user."""
    user_id: str
    created_at: datetime = field(default_factory=datetime.now)

    # Neutral baseline (always required)
    neutral_embedding: Optional[np.ndarray] = None  # 1280-dim
    neutral_valence: float = 0.0
    neutral_arousal: float = 0.0

    # Happy baseline
    happy_embedding: Optional[np.ndarray] = None  # 1280-dim
    happy_valence: float = 0.0
    happy_arousal: float = 0.0

    # Calm baseline
    calm_embedding: Optional[np.ndarray] = None  # 1280-dim
    calm_valence: float = 0.0
    calm_arousal: float = 0.0

    def is_complete(self) -> bool:
        """Check if minimum calibration requirements are met."""
        return (
            self.neutral_embedding is not None and
            self.happy_embedding is not None
        )


# ============================================================================
# HSEmotion Extractor
# ============================================================================

class HSEmotionExtractor:
    """Extract embeddings and V-A from HSEmotion (EmotiEffLib)."""

    EMOTION_NAMES = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

    def __init__(self, model_name: str = 'enet_b0_8_va_mtl'):
        self.model_name = model_name
        self.model = None
        self.is_mtl = '_mtl' in model_name

    def load(self, status_callback=None):
        """Load the HSEmotion model."""
        if status_callback:
            status_callback(f"Loading {self.model_name}...")

        from emotiefflib.facial_analysis import EmotiEffLibRecognizer
        self.model = EmotiEffLibRecognizer(
            engine='onnx',
            model_name=self.model_name
        )

        if status_callback:
            status_callback("Model loaded!")

    def extract(self, face_image: np.ndarray) -> Dict:
        """
        Extract embedding, emotion probs, and V-A from a face image.

        Args:
            face_image: RGB face image as numpy array (H, W, 3)

        Returns:
            dict with keys: embedding, emotion_probs, valence, arousal, top_emotion, confidence
        """
        if self.model is None:
            self.load()

        # Extract 1280-dim embedding
        embedding = self.model.extract_features(face_image)
        embedding = embedding[0]  # Remove batch dimension -> (1280,)

        # Get emotion predictions
        _, scores = self.model.predict_emotions(face_image, logits=False)
        scores = scores[0]  # Remove batch dimension

        # Parse results
        if self.is_mtl:
            # Last 2 values are valence and arousal
            emotion_probs = {name: float(scores[i]) for i, name in enumerate(self.EMOTION_NAMES)}
            valence = float(scores[-2])
            arousal = float(scores[-1])
        else:
            emotion_probs = {name: float(scores[i]) for i, name in enumerate(self.EMOTION_NAMES)}
            valence = 0.0
            arousal = 0.0

        # Find top emotion
        top_emotion = max(emotion_probs, key=emotion_probs.get)
        confidence = emotion_probs[top_emotion]

        return {
            'embedding': embedding,
            'emotion_probs': emotion_probs,
            'valence': valence,
            'arousal': arousal,
            'top_emotion': top_emotion,
            'confidence': confidence
        }


# ============================================================================
# Calibration Manager
# ============================================================================

class CalibrationManager:
    """Save/load user profiles to pickle files."""

    def __init__(self, storage_dir: str = './user_profiles'):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def save_profile(self, baseline: UserBaseline) -> str:
        """Save user profile to disk. Returns filepath."""
        filepath = os.path.join(self.storage_dir, f"{baseline.user_id}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(baseline, f)
        return filepath

    def load_profile(self, user_id: str) -> Optional[UserBaseline]:
        """Load user profile from disk."""
        filepath = os.path.join(self.storage_dir, f"{user_id}.pkl")
        if not os.path.exists(filepath):
            return None

        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def has_profile(self, user_id: str) -> bool:
        """Check if user has a saved profile."""
        filepath = os.path.join(self.storage_dir, f"{user_id}.pkl")
        return os.path.exists(filepath)

    def list_profiles(self) -> List[str]:
        """List all saved user IDs."""
        profiles = []
        for fname in os.listdir(self.storage_dir):
            if fname.endswith('.pkl'):
                profiles.append(fname[:-4])  # Remove .pkl extension
        return profiles

    def delete_profile(self, user_id: str) -> bool:
        """Delete a user profile."""
        filepath = os.path.join(self.storage_dir, f"{user_id}.pkl")
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False


# ============================================================================
# Calibrated Detector
# ============================================================================

class CalibratedDetector:
    """Compare current state to user's baseline for calibrated predictions."""

    # Quadrant descriptions
    QUADRANT_LABELS = {
        'Q1': 'Q1 (Happy/Excited)',
        'Q2': 'Q2 (Angry/Anxious)',
        'Q3': 'Q3 (Sad/Depressed)',
        'Q4': 'Q4 (Calm/Content)',
        'Neutral': 'Neutral'
    }

    def __init__(self):
        self.baseline: Optional[UserBaseline] = None
        self.similarity_threshold: float = 0.80
        self.neutral_threshold: float = 0.85
        self.raw_override_confidence: float = 0.60
        self.deviation_floor: float = 0.60
        self.calibrated_emotions: set = {'Happiness', 'Neutral'}

        # V-A shift thresholds (adaptive from calibration variance)
        self.va_strong_threshold: float = -0.25   # overrides calibration
        self.va_moderate_threshold: float = -0.15  # fires after calibration miss

    def set_baseline(self, baseline: UserBaseline):
        """Set the user baseline for calibrated predictions."""
        self.baseline = baseline

    def set_adaptive_thresholds(self, thresholds: Dict):
        """Apply adaptive thresholds from compute_adaptive_thresholds()."""
        self.similarity_threshold = thresholds['similarity_threshold']
        self.neutral_threshold = thresholds['neutral_threshold']
        self.deviation_floor = thresholds['deviation_floor']
        self.raw_override_confidence = thresholds['raw_override_confidence']
        if 'va_strong_threshold' in thresholds:
            self.va_strong_threshold = thresholds['va_strong_threshold']
        if 'va_moderate_threshold' in thresholds:
            self.va_moderate_threshold = thresholds['va_moderate_threshold']

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _va_to_quadrant(self, valence: float, arousal: float, threshold: float = 0.1) -> str:
        """Map V-A coordinates to quadrant."""
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

    def get_raw_prediction(self, extraction_result: Dict) -> Dict:
        """
        Format raw model output for display.

        Args:
            extraction_result: Output from HSEmotionExtractor.extract()

        Returns:
            Dict with formatted raw prediction data
        """
        valence = extraction_result['valence']
        arousal = extraction_result['arousal']

        return {
            'emotion': extraction_result['top_emotion'],
            'confidence': extraction_result['confidence'],
            'valence': valence,
            'arousal': arousal,
            'quadrant': self._va_to_quadrant(valence, arousal),
            'quadrant_label': self.QUADRANT_LABELS[self._va_to_quadrant(valence, arousal)],
            'emotion_probs': extraction_result.get('emotion_probs', {}),
        }

    def get_calibrated_prediction(self, extraction_result: Dict) -> Dict:
        """
        Get calibrated prediction using user's baseline.

        Args:
            extraction_result: Output from HSEmotionExtractor.extract()

        Returns:
            Dict with calibrated prediction data including similarities
        """
        if self.baseline is None or not self.baseline.is_complete():
            # No calibration - return raw with warning
            raw = self.get_raw_prediction(extraction_result)
            raw['warning'] = 'No calibration available'
            raw['calibrated'] = False
            return raw

        current_embedding = extraction_result['embedding']
        current_valence = extraction_result['valence']
        current_arousal = extraction_result['arousal']

        # Compute similarities to baseline states
        similarities = {}
        if self.baseline.neutral_embedding is not None:
            similarities['neutral'] = self._cosine_similarity(current_embedding, self.baseline.neutral_embedding)
        if self.baseline.happy_embedding is not None:
            similarities['happy'] = self._cosine_similarity(current_embedding, self.baseline.happy_embedding)
        if self.baseline.calm_embedding is not None:
            similarities['calm'] = self._cosine_similarity(current_embedding, self.baseline.calm_embedding)

        # Find closest baseline state
        closest_state = max(similarities, key=similarities.get)
        closest_similarity = similarities[closest_state]

        # Compute V-A shift from neutral baseline
        valence_shift = current_valence - self.baseline.neutral_valence
        arousal_shift = current_arousal - self.baseline.neutral_arousal

        # Determine calibrated quadrant from shifted V-A
        calibrated_quadrant = self._va_to_quadrant(valence_shift, arousal_shift)

        # Decision logic (calibration holds priority)
        raw_emotion = extraction_result['top_emotion']
        raw_confidence = extraction_result['confidence']
        below_deviation_floor = closest_similarity < self.deviation_floor
        emotion_probs = extraction_result.get('emotion_probs', {})

        def _best_non_cal():
            non_cal = {k: v for k, v in emotion_probs.items()
                       if k not in self.calibrated_emotions}
            if non_cal and max(non_cal.values()) > 0.05:
                return max(non_cal, key=non_cal.get)
            return None

        # Rule 0: Strong V-A shift overrides everything (clearly negative face)
        # This prevents calibration from suppressing obvious sadness.
        if calibrated_quadrant == 'Q3' and valence_shift < self.va_strong_threshold:
            calibrated_emotion = 'Sad'
            emotion_source = 'va_override'

        # Rule 1: Closest baseline passes threshold → use calibration
        elif closest_state == 'happy' and closest_similarity > self.similarity_threshold:
            calibrated_emotion = 'Happy'
            emotion_source = 'calibration'
        elif closest_state == 'neutral' and closest_similarity > self.neutral_threshold:
            calibrated_emotion = 'Neutral'
            emotion_source = 'calibration'

        # Rule 2: Raw model confidently detects non-calibrated emotion
        elif raw_emotion not in self.calibrated_emotions and raw_confidence > self.raw_override_confidence:
            calibrated_emotion = raw_emotion
            emotion_source = 'raw_model'

        # Rule 2b: Moderate V-A shift (fires when calibration didn't match)
        elif calibrated_quadrant == 'Q3' and valence_shift < self.va_moderate_threshold:
            calibrated_emotion = 'Sad'
            emotion_source = 'va_shift'

        # Rule 3: Below deviation floor → user in uncalibrated territory
        elif below_deviation_floor:
            best = _best_non_cal()
            if best:
                calibrated_emotion = best
                emotion_source = 'deviation_fallback'
            else:
                calibrated_emotion = raw_emotion
                emotion_source = 'raw_model'

        # Rule 4: Raw says calibrated emotion but rejected → use best non-cal
        elif raw_emotion in self.calibrated_emotions:
            best = _best_non_cal()
            if best:
                calibrated_emotion = best
                emotion_source = 'fallback'
            else:
                calibrated_emotion = raw_emotion
                emotion_source = 'raw_model'

        # Rule 5: Otherwise → raw model
        else:
            calibrated_emotion = raw_emotion
            emotion_source = 'raw_model'

        # Compute calibrated confidence based on source
        if emotion_source == 'calibration':
            # High similarity to baseline - use similarity as confidence
            calibrated_confidence = closest_similarity
        elif emotion_source in ('va_shift', 'va_override'):
            # V-A shift detected sadness - confidence based on how strong the shift is
            calibrated_confidence = min(0.7 + abs(valence_shift) * 0.5, 0.95)
        else:
            # Falling back to raw model - use raw model's confidence
            calibrated_confidence = raw_confidence

        calibrated_confidence = max(0.0, min(1.0, calibrated_confidence))

        return {
            'calibrated': True,
            'emotion': calibrated_emotion,
            'confidence': calibrated_confidence,
            'emotion_source': emotion_source,  # 'calibration', 'va_shift', or 'raw_model'
            'valence_shift': valence_shift,
            'arousal_shift': arousal_shift,
            'quadrant': calibrated_quadrant,
            'quadrant_label': self.QUADRANT_LABELS[calibrated_quadrant],
            'similarities': similarities,
            'closest_baseline': closest_state,
            'raw_valence': current_valence,
            'raw_arousal': current_arousal
        }


# ============================================================================
# Utility Functions
# ============================================================================

def average_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
    """Average a list of embeddings into a single centroid."""
    if not embeddings:
        raise ValueError("Cannot average empty list of embeddings")
    return np.mean(np.stack(embeddings), axis=0)


def average_values(values: List[float]) -> float:
    """Average a list of float values."""
    if not values:
        return 0.0
    return float(np.mean(values))
