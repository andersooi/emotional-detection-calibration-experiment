# HSEmotion Calibration Test Prototype

Tests the calibration layer with HSEmotion, showing **raw model output vs calibrated output** side-by-side.

## Quick Start

```bash
cd calibration_test

# Use the virtual environment (already set up)
./venv/bin/python calibration_demo.py

# Or activate venv first
source venv/bin/activate
python calibration_demo.py
```

## Requirements

A virtual environment is included with all dependencies installed:

- Python 3.8+
- EmotiEffLib (HSEmotion)
- OpenCV
- Pillow
- NumPy

To recreate the venv if needed:
```bash
python3 -m venv venv
./venv/bin/pip install numpy opencv-python Pillow emotiefflib
```

## Usage

### 1. Calibration Mode

1. Click **"Calibrate"** button
2. Enter a user ID (e.g., `test_user`)
3. Follow the on-screen prompts:
   - **Neutral** (5 sec): Relaxed, natural expression
   - **Happy** (5 sec): Think of a happy memory, smile naturally
   - **Calm** (5 sec): Deep breath, relaxed and peaceful
4. Click **"Save Profile"** to store calibration

### 2. Detection Mode

After calibration, the GUI shows side-by-side comparison:

| RAW OUTPUT | CALIBRATED OUTPUT |
|------------|-------------------|
| Direct model prediction | Adjusted using your baselines |
| Absolute V-A values | V-A shifts from YOUR neutral |
| Model's quadrant | Personalized quadrant |

### 3. Loading Profiles

Click **"Load Profile"** to load a previously saved calibration.

## File Structure

```
calibration_test/
├── calibration_demo.py    # Main GUI application
├── calibration_core.py    # Calibration logic
├── user_profiles/         # Saved calibration profiles (.pkl)
└── README.md              # This file
```

## How Calibration Works

1. **Baseline Capture**: Averages 20-25 frames (last 3-4 sec of 5 sec window) for each state
2. **Embedding Storage**: Stores 1280-dim HSEmotion embeddings for neutral, happy, calm
3. **V-A Anchoring**: Stores your personal V-A coordinates for neutral state
4. **Detection**: Compares live embeddings to baselines using cosine similarity
5. **Adjustment**: Computes V-A shift relative to YOUR neutral (not model's zero)

## Interpreting Results

### Baseline Similarities
- Values 0.85+ indicate strong match to that baseline
- The closest baseline (`*` marker) influences calibrated prediction

### V-A Shift
- **V-shift**: Valence change from YOUR neutral (+ = more positive)
- **A-shift**: Arousal change from YOUR neutral (+ = more activated)

### Quadrants (Russell's Circumplex)
- **Q1**: High arousal + Positive valence (Happy/Excited)
- **Q2**: High arousal + Negative valence (Angry/Anxious)
- **Q3**: Low arousal + Negative valence (Sad/Depressed)
- **Q4**: Low arousal + Positive valence (Calm/Content)
- **Neutral**: Center (minimal shift)

## Expected Behavior

The calibrated output should:
- Better distinguish YOUR subtle expressions
- Reduce false "neutral" classifications
- Show more stable predictions for your natural expressions
- Correctly identify when you shift from your baseline state
