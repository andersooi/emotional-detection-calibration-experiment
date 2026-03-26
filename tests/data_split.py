"""
RAVDESS actor-based train/test split utility.

Splits by actor (speaker-independent) to prevent learning actor-specific
patterns. 19 train actors (~80%), 5 test actors (~20%).

RAVDESS filename format: MM-AF-EE-EI-SS-RR-AA.mp4
  MM = Modality (01=full-AV, 02=video-only, 03=audio-only)
  AF = Vocal channel (01=speech, 02=song)
  EE = Emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)
  EI = Emotional intensity (01=normal, 02=strong)
  SS = Statement (01="Kids are talking...", 02="Dogs are sitting...")
  RR = Repetition (01=1st, 02=2nd)
  AA = Actor (01-24, odd=male, even=female)

Usage:
    from tests.data_split import split_dataset, RAVDESS_EMOTIONS, TEST_ACTORS
    train_files, test_files = split_dataset('/path/to/ravdess/videos')
"""

import os
from collections import defaultdict
from typing import List, Tuple, Dict


# Fixed test actors for reproducibility (5 actors ≈ 20%)
# Selected to balance gender (2 male odd, 3 female even) and coverage
TEST_ACTORS = {2, 5, 14, 15, 21}
TRAIN_ACTORS = set(range(1, 25)) - TEST_ACTORS

# RAVDESS emotion codes → shared labels
RAVDESS_EMOTIONS = {
    '01': 'Neutral',
    '02': 'Neutral',   # Calm → Neutral
    '03': 'Happy',
    '04': 'Sad',
    '05': 'Angry',
    '06': 'Fear',
    '07': 'Disgust',
    '08': 'Surprise',
}


def parse_ravdess_filename(fname: str) -> Dict:
    """Parse RAVDESS filename into components."""
    parts = fname.replace('.mp4', '').replace('.wav', '').split('-')
    if len(parts) < 7:
        return None
    emotion_code = parts[2]
    if emotion_code not in RAVDESS_EMOTIONS:
        return None
    return {
        'modality': parts[0],
        'vocal_channel': parts[1],
        'emotion_code': emotion_code,
        'emotion': RAVDESS_EMOTIONS[emotion_code],
        'intensity': parts[3],
        'statement': parts[4],
        'repetition': parts[5],
        'actor': int(parts[6]),
        'filename': fname,
    }


def split_dataset(video_dir: str, speech_only: bool = True) -> Tuple[List[Dict], List[Dict]]:
    """Split RAVDESS dataset by actor into train/test.

    Args:
        video_dir: Directory containing RAVDESS video files (can be flat or Actor_XX subdirs)
        speech_only: If True, only include speech clips (vocal_channel='01'), not song

    Returns:
        (train_clips, test_clips) — each is a list of dicts from parse_ravdess_filename
        with 'filepath' added.
    """
    train_clips = []
    test_clips = []

    # Handle both flat directory and Actor_XX subdirectory structure
    video_files = []
    for root, dirs, files in os.walk(video_dir):
        for fname in files:
            if fname.endswith('.mp4'):
                video_files.append((os.path.join(root, fname), fname))

    for filepath, fname in sorted(video_files):
        parsed = parse_ravdess_filename(fname)
        if parsed is None:
            continue

        # Filter to speech only (skip song)
        if speech_only and parsed['vocal_channel'] != '01':
            continue

        parsed['filepath'] = filepath

        if parsed['actor'] in TEST_ACTORS:
            test_clips.append(parsed)
        else:
            train_clips.append(parsed)

    return train_clips, test_clips


def print_split_stats(train_clips: List[Dict], test_clips: List[Dict]):
    """Print statistics about the train/test split."""
    print(f"Train: {len(train_clips)} clips from {len(set(c['actor'] for c in train_clips))} actors")
    print(f"Test:  {len(test_clips)} clips from {len(set(c['actor'] for c in test_clips))} actors")
    print(f"Test actors: {sorted(set(c['actor'] for c in test_clips))}")

    print(f"\n{'Emotion':<12} {'Train':>8} {'Test':>8} {'Total':>8}")
    print("-" * 40)
    emotions = sorted(set(c['emotion'] for c in train_clips + test_clips))
    for em in emotions:
        tr = sum(1 for c in train_clips if c['emotion'] == em)
        te = sum(1 for c in test_clips if c['emotion'] == em)
        print(f"{em:<12} {tr:>8} {te:>8} {tr+te:>8}")
    print("-" * 40)
    print(f"{'Total':<12} {len(train_clips):>8} {len(test_clips):>8} {len(train_clips)+len(test_clips):>8}")


if __name__ == "__main__":
    import sys
    video_dir = sys.argv[1] if len(sys.argv) > 1 else "data/ravdess"
    if not os.path.exists(video_dir):
        print(f"Directory not found: {video_dir}")
        print("Download RAVDESS dataset first.")
        sys.exit(1)
    train, test = split_dataset(video_dir)
    print_split_stats(train, test)
