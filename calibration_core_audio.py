"""
Self-contained audio calibration logic for Emotion2Vec testing.

This module provides:
- AudioUserBaseline: Dataclass storing audio calibration data
- Emotion2VecExtractor: Extract embeddings and emotions from audio
- AudioCalibrationManager: Save/load audio profiles
- CalibratedAudioDetector: Compare raw vs calibrated predictions
"""

import os
import pickle
import tempfile
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple, List


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class AudioUserBaseline:
    """Stores audio calibration data for one user."""
    user_id: str
    created_at: datetime = field(default_factory=datetime.now)

    # Neutral baseline (always required)
    neutral_embedding: Optional[np.ndarray] = None  # 768-dim

    # Happy baseline
    happy_embedding: Optional[np.ndarray] = None  # 768-dim

    # Calm baseline
    calm_embedding: Optional[np.ndarray] = None  # 768-dim

    def is_complete(self) -> bool:
        """Check if minimum calibration requirements are met."""
        return (
            self.neutral_embedding is not None and
            self.happy_embedding is not None and
            self.calm_embedding is not None
        )


# ============================================================================
# Emotion2Vec Extractor
# ============================================================================

class Emotion2VecExtractor:
    """Extract embeddings and emotions from audio using Emotion2Vec."""

    # Emotion labels from Emotion2Vec (bilingual -> English)
    EMOTION_MAP = {
        '生气/angry': 'Angry',
        '厌恶/disgusted': 'Disgust',
        '恐惧/fearful': 'Fear',
        '开心/happy': 'Happy',
        '中立/neutral': 'Neutral',
        '其他/other': 'Other',
        '难过/sad': 'Sad',
        '吃惊/surprised': 'Surprise',
        '<unk>': 'Unknown'
    }

    def __init__(self, model_size: str = 'base'):
        self.model_size = model_size
        self.model = None

    def load(self, status_callback=None):
        """Load the Emotion2Vec model."""
        if status_callback:
            status_callback(f"Loading Emotion2Vec ({self.model_size})...")

        from funasr import AutoModel
        model_name = f"iic/emotion2vec_plus_{self.model_size}"
        self.model = AutoModel(model=model_name, hub='hf', disable_update=True)

        if status_callback:
            status_callback("Emotion2Vec loaded!")

    def extract(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict:
        """
        Extract embedding and emotion probs from audio.

        Args:
            audio: Audio waveform as numpy array (1D, float32)
            sample_rate: Audio sample rate (default 16000)

        Returns:
            dict with keys: embedding, emotion_probs, top_emotion, confidence
        """
        if self.model is None:
            self.load()

        import soundfile as sf

        # Ensure audio is float32 and normalized
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        # Save to temp file (FunASR requires file path)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio, sample_rate)

        try:
            # Extract with embeddings
            result = self.model.generate(
                temp_path,
                granularity='utterance',
                extract_embedding=True
            )

            # Parse results
            output = result[0]
            embedding = output['feats']  # Shape (768,)
            labels = output['labels']
            scores = np.array(output['scores'])

            # Normalize scores to probabilities
            if scores.sum() > 0:
                scores = scores / scores.sum()

            # Create emotion dict with clean names
            emotion_probs = {}
            for label, score in zip(labels, scores):
                clean_name = self.EMOTION_MAP.get(label, label)
                emotion_probs[clean_name] = float(score)

            # Find top emotion
            top_emotion = max(emotion_probs, key=emotion_probs.get)
            confidence = emotion_probs[top_emotion]

            return {
                'embedding': embedding,
                'emotion_probs': emotion_probs,
                'top_emotion': top_emotion,
                'confidence': confidence
            }

        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass


# ============================================================================
# Audio Calibration Manager
# ============================================================================

class AudioCalibrationManager:
    """Save/load audio user profiles to pickle files."""

    def __init__(self, storage_dir: str = './user_profiles'):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def save_profile(self, baseline: AudioUserBaseline) -> str:
        """Save user profile to disk. Returns filepath."""
        filepath = os.path.join(self.storage_dir, f"{baseline.user_id}_audio.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(baseline, f)
        return filepath

    def load_profile(self, user_id: str) -> Optional[AudioUserBaseline]:
        """Load user profile from disk."""
        filepath = os.path.join(self.storage_dir, f"{user_id}_audio.pkl")
        if not os.path.exists(filepath):
            return None

        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def has_profile(self, user_id: str) -> bool:
        """Check if user has a saved audio profile."""
        filepath = os.path.join(self.storage_dir, f"{user_id}_audio.pkl")
        return os.path.exists(filepath)

    def list_profiles(self) -> List[str]:
        """List all saved audio user IDs."""
        profiles = []
        for fname in os.listdir(self.storage_dir):
            if fname.endswith('_audio.pkl'):
                profiles.append(fname[:-10])  # Remove _audio.pkl
        return profiles

    def delete_profile(self, user_id: str) -> bool:
        """Delete an audio user profile."""
        filepath = os.path.join(self.storage_dir, f"{user_id}_audio.pkl")
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False


# ============================================================================
# Calibrated Audio Detector
# ============================================================================

class CalibratedAudioDetector:
    """Compare current audio to user's baseline for calibrated predictions."""

    def __init__(self):
        self.baseline: Optional[AudioUserBaseline] = None

    def set_baseline(self, baseline: AudioUserBaseline):
        """Set the user baseline for calibrated predictions."""
        self.baseline = baseline

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_raw_prediction(self, extraction_result: Dict) -> Dict:
        """
        Format raw model output for display.

        Args:
            extraction_result: Output from Emotion2VecExtractor.extract()

        Returns:
            Dict with formatted raw prediction data
        """
        return {
            'emotion': extraction_result['top_emotion'],
            'confidence': extraction_result['confidence'],
            'emotion_probs': extraction_result['emotion_probs']
        }

    def get_calibrated_prediction(self, extraction_result: Dict) -> Dict:
        """
        Get calibrated prediction using user's baseline.

        Args:
            extraction_result: Output from Emotion2VecExtractor.extract()

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

        # Compute similarities to baseline states
        sim_neutral = self._cosine_similarity(current_embedding, self.baseline.neutral_embedding)
        sim_happy = self._cosine_similarity(current_embedding, self.baseline.happy_embedding)
        sim_calm = self._cosine_similarity(current_embedding, self.baseline.calm_embedding)

        similarities = {
            'neutral': sim_neutral,
            'happy': sim_happy,
            'calm': sim_calm
        }

        # Find closest baseline state
        closest_state = max(similarities, key=similarities.get)
        closest_similarity = similarities[closest_state]

        # Determine calibrated emotion using hybrid approach
        raw_emotion = extraction_result['top_emotion']
        raw_confidence = extraction_result['confidence']

        # Thresholds (same as visual for now, can tune later)
        HAPPY_CALM_THRESHOLD = 0.80
        NEUTRAL_THRESHOLD = 0.85

        if closest_state == 'happy' and closest_similarity > HAPPY_CALM_THRESHOLD:
            calibrated_emotion = 'Happy'
            emotion_source = 'calibration'
        elif closest_state == 'calm' and closest_similarity > HAPPY_CALM_THRESHOLD:
            calibrated_emotion = 'Calm'
            emotion_source = 'calibration'
        elif closest_state == 'neutral' and closest_similarity > NEUTRAL_THRESHOLD:
            calibrated_emotion = 'Neutral'
            emotion_source = 'calibration'
        else:
            # Fall back to raw model's prediction
            calibrated_emotion = raw_emotion
            emotion_source = 'raw_model'

        # Compute calibrated confidence
        if emotion_source == 'calibration':
            calibrated_confidence = closest_similarity
        else:
            calibrated_confidence = raw_confidence

        calibrated_confidence = max(0.0, min(1.0, calibrated_confidence))

        return {
            'calibrated': True,
            'emotion': calibrated_emotion,
            'confidence': calibrated_confidence,
            'emotion_source': emotion_source,
            'similarities': similarities,
            'closest_baseline': closest_state
        }


# ============================================================================
# Utility Functions
# ============================================================================

def average_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
    """Average a list of embeddings into a single centroid."""
    if not embeddings:
        raise ValueError("Cannot average empty list of embeddings")
    return np.mean(np.stack(embeddings), axis=0)
