"""
Generalised calibration interface for any emotion detection model.

Provides abstract base classes that any model can implement to plug into
the calibration and fusion pipeline. Team members just need to subclass
BaseEmotionExtractor and implement load() + extract().

Usage:
    from calibration_base import BaseEmotionExtractor, GenericBaseline,
        GenericCalibrationManager, GenericCalibratedDetector

    # 1. Implement extractor for your model
    class MyModelExtractor(BaseEmotionExtractor):
        def load(self, status_callback=None):
            self.model = load_my_model()

        def extract(self, input_data) -> Dict:
            embedding = self.model.get_features(input_data)
            probs = self.model.classify(input_data)
            return {
                'embedding': embedding,
                'emotion_probs': probs,
                'top_emotion': max(probs, key=probs.get),
                'confidence': max(probs.values())
            }

    # 2. Use with generic calibration (everything else works automatically)
    extractor = MyModelExtractor()
    manager = GenericCalibrationManager(modality='my_model')
    detector = GenericCalibratedDetector()
"""

import os
import pickle
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List, Tuple


# ============================================================================
# Abstract Extractor
# ============================================================================

class BaseEmotionExtractor(ABC):
    """
    Abstract base class for any emotion detection model.

    Subclasses must implement load() and extract().
    extract() must return a dict with at minimum:
        - embedding: np.ndarray (1D, any dimension)
        - emotion_probs: Dict[str, float]
        - top_emotion: str
        - confidence: float

    Optional keys:
        - valence: float (if model provides V-A)
        - arousal: float (if model provides V-A)
    """

    @abstractmethod
    def load(self, status_callback=None):
        """Load the model. Called once before extract()."""
        pass

    @abstractmethod
    def extract(self, input_data) -> Dict:
        """
        Run inference on input data.

        Args:
            input_data: Model-specific input (image, audio array, etc.)

        Returns:
            Dict with keys:
                embedding: np.ndarray - Feature vector (any dimension)
                emotion_probs: Dict[str, float] - Emotion label → probability
                top_emotion: str - Highest-scoring emotion
                confidence: float - Score of top emotion (0-1)
                valence: float (optional) - Valence value if model provides
                arousal: float (optional) - Arousal value if model provides
        """
        pass

    def has_va(self) -> bool:
        """Whether this model provides native valence-arousal output."""
        return False


# ============================================================================
# Generic Baseline
# ============================================================================

@dataclass
class GenericBaseline:
    """
    Model-agnostic baseline storage.

    Works with any embedding dimension and any number of calibration states.
    """
    user_id: str
    modality: str  # e.g., 'face', 'audio', 'gaze'
    created_at: datetime = field(default_factory=datetime.now)

    # Embeddings for each calibration state (state_name → embedding)
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)

    # Optional V-A values per state (state_name → (valence, arousal))
    va_values: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Optional metadata
    metadata: Dict = field(default_factory=dict)

    # Default calibration states
    REQUIRED_STATES = ['neutral', 'happy', 'calm']

    def add_state(self, state: str, embedding: np.ndarray,
                  valence: Optional[float] = None, arousal: Optional[float] = None):
        """Add a calibration state."""
        self.embeddings[state] = embedding
        if valence is not None and arousal is not None:
            self.va_values[state] = (valence, arousal)

    def get_embedding(self, state: str) -> Optional[np.ndarray]:
        """Get embedding for a state."""
        return self.embeddings.get(state)

    def get_va(self, state: str) -> Optional[Tuple[float, float]]:
        """Get V-A for a state (if available)."""
        return self.va_values.get(state)

    def is_complete(self) -> bool:
        """Check if all required states have been captured."""
        return all(state in self.embeddings for state in self.REQUIRED_STATES)

    def get_states(self) -> List[str]:
        """Get list of captured states."""
        return list(self.embeddings.keys())

    def embedding_dim(self) -> Optional[int]:
        """Get embedding dimensionality (from first stored embedding)."""
        if self.embeddings:
            first = next(iter(self.embeddings.values()))
            return first.shape[0]
        return None


# ============================================================================
# Generic Calibration Manager
# ============================================================================

class GenericCalibrationManager:
    """
    Model-agnostic profile persistence.

    Stores profiles as {user_id}_{modality}.pkl to avoid collisions
    between different modalities for the same user.
    """

    def __init__(self, modality: str, storage_dir: str = './user_profiles'):
        self.modality = modality
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def _filepath(self, user_id: str) -> str:
        return os.path.join(self.storage_dir, f"{user_id}_{self.modality}.pkl")

    def save_profile(self, baseline: GenericBaseline) -> str:
        """Save profile to disk. Returns filepath."""
        filepath = self._filepath(baseline.user_id)
        with open(filepath, 'wb') as f:
            pickle.dump(baseline, f)
        return filepath

    def load_profile(self, user_id: str) -> Optional[GenericBaseline]:
        """Load profile from disk."""
        filepath = self._filepath(user_id)
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def has_profile(self, user_id: str) -> bool:
        filepath = self._filepath(user_id)
        return os.path.exists(filepath)

    def list_profiles(self) -> List[str]:
        suffix = f"_{self.modality}.pkl"
        profiles = []
        for fname in os.listdir(self.storage_dir):
            if fname.endswith(suffix):
                profiles.append(fname[:-len(suffix)])
        return profiles

    def delete_profile(self, user_id: str) -> bool:
        filepath = self._filepath(user_id)
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False


# ============================================================================
# Generic Calibrated Detector
# ============================================================================

class GenericCalibratedDetector:
    """
    Model-agnostic calibrated emotion detector.

    Works with any embedding dimension. Uses cosine similarity to compare
    live embeddings against calibration baselines.

    Configurable thresholds and override behaviour.
    """

    def __init__(
        self,
        calibrated_emotions: Optional[set] = None,
        similarity_threshold: float = 0.80,
        neutral_threshold: float = 0.85,
        raw_override_confidence: float = 0.60
    ):
        """
        Args:
            calibrated_emotions: Set of emotion labels we have baselines for.
                Defaults to {'Happy', 'Neutral', 'Calm'}.
            similarity_threshold: Cosine similarity threshold for non-neutral states.
            neutral_threshold: Stricter threshold for neutral state.
            raw_override_confidence: If raw model detects a non-calibrated emotion
                above this confidence, trust raw model.
        """
        self.baseline: Optional[GenericBaseline] = None
        self.calibrated_emotions = calibrated_emotions or {'Happy', 'Neutral', 'Calm'}
        self.similarity_threshold = similarity_threshold
        self.neutral_threshold = neutral_threshold
        self.raw_override_confidence = raw_override_confidence

    def set_baseline(self, baseline: GenericBaseline):
        self.baseline = baseline

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_raw_prediction(self, extraction_result: Dict) -> Dict:
        """Format raw model output."""
        return {
            'emotion': extraction_result['top_emotion'],
            'confidence': extraction_result['confidence'],
            'valence': extraction_result.get('valence'),
            'arousal': extraction_result.get('arousal'),
            'emotion_probs': extraction_result.get('emotion_probs', {})
        }

    def get_calibrated_prediction(self, extraction_result: Dict) -> Dict:
        """
        Get calibrated prediction using stored baseline.

        Logic:
        1. If raw model confidently detects a non-calibrated emotion → trust raw
        2. If high similarity to a calibrated baseline → use calibration
        3. Otherwise → fall back to raw model
        """
        if self.baseline is None or not self.baseline.is_complete():
            raw = self.get_raw_prediction(extraction_result)
            raw['calibrated'] = False
            raw['warning'] = 'No calibration available'
            return raw

        current_embedding = extraction_result['embedding']

        # Compute similarities to all baseline states
        similarities = {}
        for state, baseline_emb in self.baseline.embeddings.items():
            similarities[state] = self._cosine_similarity(current_embedding, baseline_emb)

        closest_state = max(similarities, key=similarities.get)
        closest_similarity = similarities[closest_state]

        # Raw model output
        raw_emotion = extraction_result['top_emotion']
        raw_confidence = extraction_result['confidence']

        # Decision logic
        # Rule 1: If raw model confidently detects non-calibrated emotion, trust it
        if raw_emotion not in self.calibrated_emotions and raw_confidence > self.raw_override_confidence:
            calibrated_emotion = raw_emotion
            emotion_source = 'raw_model'
        else:
            # Rule 2: Check similarity to calibrated baselines
            if closest_state == 'neutral' and closest_similarity > self.neutral_threshold:
                calibrated_emotion = 'Neutral'
                emotion_source = 'calibration'
            elif closest_similarity > self.similarity_threshold:
                # Map state name to emotion label
                state_to_emotion = {'neutral': 'Neutral', 'happy': 'Happy', 'calm': 'Calm'}
                calibrated_emotion = state_to_emotion.get(closest_state, closest_state.title())
                emotion_source = 'calibration'
            else:
                # Rule 3: Fall back to raw
                calibrated_emotion = raw_emotion
                emotion_source = 'raw_model'

        # Confidence
        if emotion_source == 'calibration':
            calibrated_confidence = closest_similarity
        else:
            calibrated_confidence = raw_confidence

        calibrated_confidence = max(0.0, min(1.0, calibrated_confidence))

        # V-A shift (if available)
        valence_shift = None
        arousal_shift = None
        neutral_va = self.baseline.get_va('neutral')
        if neutral_va and 'valence' in extraction_result and extraction_result['valence'] is not None:
            valence_shift = extraction_result['valence'] - neutral_va[0]
            arousal_shift = extraction_result['arousal'] - neutral_va[1]

        return {
            'calibrated': True,
            'emotion': calibrated_emotion,
            'confidence': calibrated_confidence,
            'emotion_source': emotion_source,
            'similarities': similarities,
            'closest_baseline': closest_state,
            'valence_shift': valence_shift,
            'arousal_shift': arousal_shift
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


# ============================================================================
# Wrapper: Adapt Existing Extractors to Base Interface
# ============================================================================

class HSEmotionExtractorAdapter(BaseEmotionExtractor):
    """Wraps existing HSEmotionExtractor to conform to BaseEmotionExtractor."""

    def __init__(self, model_name: str = 'enet_b0_8_va_mtl'):
        self.model_name = model_name
        self._extractor = None

    def load(self, status_callback=None):
        from core.calibration_visual import HSEmotionExtractor
        self._extractor = HSEmotionExtractor(model_name=self.model_name)
        self._extractor.load(status_callback)

    def extract(self, input_data) -> Dict:
        return self._extractor.extract(input_data)

    def has_va(self) -> bool:
        return '_mtl' in self.model_name


class Emotion2VecExtractorAdapter(BaseEmotionExtractor):
    """Wraps existing Emotion2VecExtractor to conform to BaseEmotionExtractor."""

    def __init__(self, model_size: str = 'large'):
        self.model_size = model_size
        self._extractor = None

    def load(self, status_callback=None):
        from core.calibration_audio import Emotion2VecExtractor
        self._extractor = Emotion2VecExtractor(model_size=self.model_size)
        self._extractor.load(status_callback)

    def extract(self, input_data) -> Dict:
        if isinstance(input_data, tuple):
            audio, sr = input_data
            return self._extractor.extract(audio, sr)
        return self._extractor.extract(input_data)

    def has_va(self) -> bool:
        return False
