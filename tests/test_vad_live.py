"""
Standalone VAD test — run this to see when FunASR's FSMN VAD
detects speech vs silence on your live microphone.

Prints [SPEECH] or [SILENCE] every second so you can verify
it correctly distinguishes your voice from background noise.

Usage:
    venv/bin/python tests/test_vad_live.py

Press Ctrl+C to stop.
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import time
from funasr import AutoModel

SAMPLE_RATE = 16000
CHUNK_DURATION = 2.0  # seconds
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)


def main():
    print("Loading FunASR VAD model...")
    vad_model = AutoModel(
        model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        device="cpu",
    )
    print("VAD model loaded.")
    print()
    print("Listening on microphone... Speak to test. Ctrl+C to stop.")
    print("=" * 60)

    buffer = np.array([], dtype=np.float32)
    speech_count = 0
    silence_count = 0

    def audio_callback(indata, frames, time_info, status):
        nonlocal buffer
        if status:
            print(f"  (audio status: {status})")
        buffer = np.concatenate([buffer, indata[:, 0].copy()])

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype=np.float32,
        callback=audio_callback, blocksize=int(SAMPLE_RATE * 0.1))

    try:
        stream.start()
        while True:
            # Wait for enough audio
            if len(buffer) < CHUNK_SAMPLES:
                time.sleep(0.1)
                continue

            # Take a chunk
            chunk = buffer[:CHUNK_SAMPLES].copy()
            buffer = buffer[CHUNK_SAMPLES // 2:]  # 50% overlap

            # Compute RMS for comparison
            rms = float(np.sqrt(np.mean(chunk ** 2)))

            # Run VAD — needs a temp file (FunASR API)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
                sf.write(temp_path, chunk, SAMPLE_RATE)

            try:
                result = vad_model.generate(input=temp_path)
            finally:
                import os
                os.unlink(temp_path)

            # Parse VAD result — returns list of [start_ms, end_ms] segments
            segments = result[0].get('value', []) if result else []
            has_speech = len(segments) > 0

            if has_speech:
                speech_count += 1
                # Calculate speech duration
                total_speech_ms = sum(seg[1] - seg[0] for seg in segments)
                print(f"  [SPEECH]  rms={rms:.4f}  "
                      f"segments={len(segments)}  "
                      f"speech={total_speech_ms}ms / {int(CHUNK_DURATION*1000)}ms")
            else:
                silence_count += 1
                print(f"  [SILENCE] rms={rms:.4f}")

    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print(f"Done. Speech chunks: {speech_count}, Silence chunks: {silence_count}")
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
