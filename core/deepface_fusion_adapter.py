"""
Adapter that converts DeepFace calibrated predictions into a coherent
probability vector for the fusion layer.

Without this adapter, calibration only changes the display label but
doesn't affect the probability vector that fusion actually uses.
When calibration fires (Neutral or Happiness), this adapter boosts the
calibrated class in the probability vector and scales others down,
so fusion sees a distribution consistent with the calibrated decision.
"""

from typing import Dict


def build_face_result(raw_result: Dict, cal_result: Dict) -> Dict:
    """Build a face_result dict suitable for fusion from raw + calibrated outputs.

    Args:
        raw_result: From GenericCalibratedDetector.get_raw_prediction().
            Must have: emotion (or top_emotion), confidence, emotion_probs.
        cal_result: From GenericCalibratedDetector.get_calibrated_prediction().
            Must have: calibrated, emotion, confidence, emotion_source.

    Returns:
        Dict with top_emotion, confidence, emotion_probs, _face_source.
        The probability vector is coherent with the calibrated decision.
    """
    probs = dict(raw_result['emotion_probs'])
    raw_top = raw_result.get('top_emotion') or raw_result.get('emotion')
    if raw_top is None:
        raise KeyError("raw_result needs 'emotion' or 'top_emotion'")
    raw_conf = raw_result['confidence']

    # No calibration available — pass through raw
    if not cal_result.get('calibrated'):
        return {
            'top_emotion': raw_top,
            'confidence': raw_conf,
            'emotion_probs': probs,
            '_face_source': 'deepface_raw',
        }

    source = cal_result.get('emotion_source', '')
    cal_emotion = cal_result.get('emotion', raw_top)
    cal_conf = float(cal_result.get('confidence', raw_conf))

    # Calibration active but didn't override (raw_model, fallback, deviation)
    # Use the calibrated label but keep raw probabilities
    if source != 'calibration':
        return {
            'top_emotion': cal_emotion,
            'confidence': cal_conf if source not in ('raw_model',) else raw_conf,
            'emotion_probs': probs,
            '_face_source': f'deepface_{source}',
        }

    # Calibration fired — boost target class in probability vector
    # Map calibrated label back to DeepFace's label space
    target = {'Happy': 'Happiness', 'Neutral': 'Neutral'}.get(cal_emotion, cal_emotion)

    # Set target probability: at least what raw gave, up to calibration confidence (capped at 0.90)
    target_prob = max(probs.get(target, 0.0), min(cal_conf, 0.90))

    # Scale other classes proportionally so everything sums to 1.0
    other_total = sum(v for k, v in probs.items() if k != target)
    if other_total > 0:
        scale = max(0.0, 1.0 - target_prob) / other_total
        for k in list(probs.keys()):
            if k != target:
                probs[k] *= scale
    probs[target] = target_prob

    return {
        'top_emotion': target,
        'confidence': target_prob,
        'emotion_probs': probs,
        '_face_source': 'deepface_calibration',
    }
