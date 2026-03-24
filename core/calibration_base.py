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
        raw_override_confidence: float = 0.60,
        deviation_floor: float = 0.60,
    ):
        """
        Args:
            calibrated_emotions: Set of emotion labels we have baselines for.
                Defaults to {'Happy', 'Neutral', 'Calm'}.
            similarity_threshold: Cosine similarity threshold for non-neutral states.
            neutral_threshold: Stricter threshold for neutral state.
            raw_override_confidence: If raw model detects a non-calibrated emotion
                above this confidence, trust raw model.
            deviation_floor: If max similarity to ALL baselines is below this,
                user has deviated into uncalibrated territory.
        """
        self.baseline: Optional[GenericBaseline] = None
        self.calibrated_emotions = calibrated_emotions or {'Happy', 'Neutral', 'Calm'}
        self.similarity_threshold = similarity_threshold
        self.neutral_threshold = neutral_threshold
        self.raw_override_confidence = raw_override_confidence
        self.deviation_floor = deviation_floor

    def set_baseline(self, baseline: GenericBaseline):
        self.baseline = baseline

    def set_adaptive_thresholds(self, thresholds: Dict):
        """Apply adaptive thresholds from compute_adaptive_thresholds()."""
        self.similarity_threshold = thresholds['similarity_threshold']
        self.neutral_threshold = thresholds['neutral_threshold']
        self.deviation_floor = thresholds['deviation_floor']
        self.raw_override_confidence = thresholds['raw_override_confidence']

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

        # Check if user has deviated far from ALL baselines
        below_deviation_floor = closest_similarity < self.deviation_floor
        emotion_probs = extraction_result.get('emotion_probs', {})

        # Helper: find best non-calibrated emotion from raw probs
        def _best_non_cal():
            non_cal = {k: v for k, v in emotion_probs.items()
                       if k not in self.calibrated_emotions}
            if non_cal and max(non_cal.values()) > 0.05:
                return max(non_cal, key=non_cal.get)
            return None

        # Decision logic: calibration has priority, but very confident raw
        # predictions require stronger calibration match to override.
        # If raw says non-calibrated emotion at >90%, boost the threshold
        # so only a strong calibration match can override it.
        raw_is_strong_noncat = (
            raw_emotion not in self.calibrated_emotions and raw_confidence > 0.90
        )
        effective_sim_threshold = self.similarity_threshold + (0.05 if raw_is_strong_noncat else 0)
        effective_neu_threshold = self.neutral_threshold + (0.05 if raw_is_strong_noncat else 0)

        # Rule 1: Closest baseline passes threshold → use calibration
        if closest_state == 'neutral' and closest_similarity > effective_neu_threshold:
            calibrated_emotion = 'Neutral'
            emotion_source = 'calibration'
        elif closest_similarity > effective_sim_threshold:
            state_to_emotion = {'neutral': 'Neutral', 'happy': 'Happy', 'calm': 'Calm'}
            calibrated_emotion = state_to_emotion.get(closest_state, closest_state.title())
            emotion_source = 'calibration'

        # Rule 2: Raw model confidently detects non-calibrated emotion
        elif raw_emotion not in self.calibrated_emotions and raw_confidence > self.raw_override_confidence:
            calibrated_emotion = raw_emotion
            emotion_source = 'raw_model'

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

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_adaptive_thresholds(
    neutral_emb: np.ndarray,
    happy_emb: np.ndarray,
    calm_emb: np.ndarray,
    margin: float = 0.03,
) -> Dict:
    """
    Compute user-adaptive similarity thresholds from calibration baselines.

    If baselines are close together (subtle expressions), thresholds are strict.
    If baselines are far apart (expressive user), thresholds are looser.

    Returns dict with similarity_threshold, neutral_threshold, deviation_floor,
    raw_override_confidence, and diagnostics.
    """
    sim_nh = cosine_similarity(neutral_emb, happy_emb)
    sim_nc = cosine_similarity(neutral_emb, calm_emb)
    sim_hc = cosine_similarity(happy_emb, calm_emb)

    max_inter_sim = max(sim_nh, sim_nc, sim_hc)
    min_inter_sim = min(sim_nh, sim_nc, sim_hc)

    # Threshold is a fraction of max_inter_sim.
    # During calibration, averaged baselines have very high inter-similarity (~0.99).
    # But live embeddings score lower (~0.85-0.90) due to movement, lighting, etc.
    # We set thresholds at 85% and 80% of the inter-baseline similarity to account
    # for this live-to-calibration gap.
    #
    # Examples:
    #   DeepFace (max_inter=0.99): threshold=0.84, floor=0.79
    #   HSEmotion (max_inter=0.92): threshold=0.78, floor=0.74
    #   Expressive (max_inter=0.80): threshold=0.70, floor=0.64
    similarity_threshold = max(0.65, min(0.95, max_inter_sim * 0.85))
    neutral_threshold = max(0.65, min(0.95, max_inter_sim * 0.86))
    deviation_floor = max(0.50, min(0.90, max_inter_sim * 0.79))

    return {
        'similarity_threshold': similarity_threshold,
        'neutral_threshold': neutral_threshold,
        'deviation_floor': deviation_floor,
        'raw_override_confidence': 0.60,
        'diagnostics': {
            'sim_neutral_happy': sim_nh,
            'sim_neutral_calm': sim_nc,
            'sim_happy_calm': sim_hc,
            'max_inter_sim': max_inter_sim,
            'min_inter_sim': min_inter_sim,
        },
    }


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


class DeepFaceExtractor(BaseEmotionExtractor):
    """
    DeepFace emotion extractor — emotion probs used as embedding for calibration.

    By default, uses the 7-dim emotion probability vector as the "embedding"
    for cosine similarity calibration. This works better than the identity
    embedding (VGG-Face 4096-dim) because the emotion classifier is trained
    to separate emotions, while the identity embedding is trained to match
    the same person regardless of expression.

    Set use_emotion_embedding=False to use the identity embedding instead.

    Usage:
        extractor = DeepFaceExtractor()
        extractor.load()
        result = extractor.extract(face_rgb_image)
    """

    def __init__(self, model_name: str = 'VGG-Face', detector_backend: str = 'skip',
                 use_emotion_embedding: bool = True):
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.use_emotion_embedding = use_emotion_embedding
        self._deepface = None

    def load(self, status_callback=None):
        if status_callback:
            status_callback(f"Loading DeepFace ({self.model_name})...")

        from deepface import DeepFace
        self._deepface = DeepFace

        # Warm up model (first call downloads weights)
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        try:
            self._deepface.analyze(dummy, actions=['emotion'],
                                   detector_backend='skip', enforce_detection=False)
            if not self.use_emotion_embedding:
                self._deepface.represent(dummy, model_name=self.model_name,
                                         detector_backend='skip', enforce_detection=False)
        except Exception:
            pass

        if status_callback:
            status_callback("DeepFace loaded!")

    def extract(self, face_image) -> Dict:
        if self._deepface is None:
            self.load()

        # Get emotion probs
        analysis = self._deepface.analyze(
            face_image, actions=['emotion'],
            detector_backend=self.detector_backend,
            enforce_detection=False
        )
        raw_probs = analysis[0]['emotion']

        # Normalize: lowercase → PascalCase, percentages (0-100) → probabilities (0-1)
        emotion_probs = {
            'Anger': raw_probs.get('angry', 0.0) / 100,
            'Disgust': raw_probs.get('disgust', 0.0) / 100,
            'Fear': raw_probs.get('fear', 0.0) / 100,
            'Happiness': raw_probs.get('happy', 0.0) / 100,
            'Neutral': raw_probs.get('neutral', 0.0) / 100,
            'Sadness': raw_probs.get('sad', 0.0) / 100,
            'Surprise': raw_probs.get('surprise', 0.0) / 100,
        }

        top_emotion = max(emotion_probs, key=emotion_probs.get)

        # Use emotion probability vector as embedding for calibration
        # This is a 7-dim vector where each dimension is an emotion probability
        # Cosine similarity on this vector compares emotion DISTRIBUTIONS
        if self.use_emotion_embedding:
            embedding = np.array(list(emotion_probs.values()))
        else:
            # Fall back to identity embedding
            rep = self._deepface.represent(
                face_image, model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            embedding = np.array(rep[0]['embedding'])

        return {
            'embedding': embedding,
            'emotion_probs': emotion_probs,
            'top_emotion': top_emotion,
            'confidence': emotion_probs[top_emotion],
        }

    def has_va(self) -> bool:
        return False


class DeepFaceEmotionEmbeddingExtractor(BaseEmotionExtractor):
    """
    Extracts 1024-dim embeddings from DeepFace's EMOTION model (not the identity model).

    DeepFace's emotion classifier is a custom CNN:
        Conv layers → Flatten → Dense(1024) → Dense(1024) → Dense(7, softmax)

    We extract the penultimate Dense(1024) output as the embedding.
    This is emotion-trained (unlike VGG-Face identity embeddings) and should
    separate emotions better for cosine similarity calibration.
    """

    def __init__(self):
        self._deepface = None
        self._emotion_model = None
        self._embedding_model = None

    def load(self, status_callback=None):
        if status_callback:
            status_callback("Loading DeepFace emotion embedding model...")

        from deepface import DeepFace
        self._deepface = DeepFace

        # Load the emotion model
        from deepface.models.demography.Emotion import EmotionClient
        client = EmotionClient()
        self._emotion_model = client.model

        # Create sub-model that outputs penultimate Dense(1024) layer
        # Architecture: ... → Dense(1024) → Dropout → Dense(1024) → Dropout → Dense(7)
        # layers[-3] is the second Dense(1024) before final Dropout and Dense(7)
        import tensorflow as tf
        self._embedding_model = tf.keras.Model(
            inputs=self._emotion_model.input,
            outputs=self._emotion_model.layers[-3].output
        )

        if status_callback:
            status_callback("DeepFace emotion embedding model loaded!")

    def _preprocess(self, face_image):
        """Preprocess face image for emotion model: RGB → grayscale → 48x48."""
        import cv2
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_image
        gray = cv2.resize(gray, (48, 48))
        return np.expand_dims(np.expand_dims(gray, axis=-1), axis=0).astype(np.float32)

    def extract(self, face_image) -> Dict:
        if self._embedding_model is None:
            self.load()

        preprocessed = self._preprocess(face_image)

        # Get 1024-dim emotion embedding
        embedding = self._embedding_model.predict(preprocessed, verbose=0)[0]

        # Get emotion probabilities from full model
        probs = self._emotion_model.predict(preprocessed, verbose=0)[0]

        # Must match DeepFace's Emotion.py label order exactly
        labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
        emotion_probs = {label: float(probs[i]) for i, label in enumerate(labels)}
        top_emotion = max(emotion_probs, key=emotion_probs.get)

        return {
            'embedding': embedding,
            'emotion_probs': emotion_probs,
            'top_emotion': top_emotion,
            'confidence': emotion_probs[top_emotion],
        }

    def has_va(self) -> bool:
        return False
