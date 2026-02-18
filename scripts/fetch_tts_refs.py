import librosa
import soundfile as sf
import os
import numpy as np


def generate_refs():
    out_dir = "assets/tts_refs"
    os.makedirs(out_dir, exist_ok=True)

    print("Downloading/Loading LibriSpeech sample...")
    # 'libri1' is a male speaker from LibriSpeech
    try:
        y, sr = librosa.load(librosa.ex('libri1'))
    except Exception as e:
        print(f"Failed to load librosa example: {e}")
        # Create a dummy silent wav if download fails (fallback)
        y = np.zeros(22050 * 3)
        sr = 22050

    # Split into segments to create "different" voices
    # (even if same speaker, at least provides valid files)

    # Narrator: 0-3s
    sf.write(os.path.join(out_dir, 'narrator.wav'), y[:sr * 3], sr)

    # Male: 3-6s
    sf.write(os.path.join(out_dir, 'male.wav'), y[sr * 3:sr * 6], sr)

    # Female: 6-9s (still male voice, placeholder — replace with real female WAV for best results)
    sf.write(os.path.join(out_dir, 'female.wav'), y[sr * 6:sr * 9], sr)

    print(f"Generated reference WAVs in {out_dir}")
    print("Tip: Replace these with real voice recordings for better character differentiation.")


if __name__ == "__main__":
    generate_refs()
