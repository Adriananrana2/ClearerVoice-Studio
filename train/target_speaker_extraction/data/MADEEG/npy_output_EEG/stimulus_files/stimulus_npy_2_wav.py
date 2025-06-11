import numpy as np
import soundfile as sf
import os

# Directory containing all .npy stimulus files
stimulus_dir = "./"

# Loop through all .npy files in the directory
for filename in os.listdir(stimulus_dir):
    if filename.endswith(".npy"):
        npy_path = os.path.join(stimulus_dir, filename)
        base_name = os.path.splitext(filename)[0]
        wav_path = os.path.join(stimulus_dir, base_name + ".wav")

        # Load the numpy array
        audio_array = np.load(npy_path)

        print(f"Processing {filename} ... Shape: {audio_array.shape}")

        # Check if it's a valid stereo file (2 channels)
        if audio_array.shape[0] != 2:
            print("  → Skipped: Not a 2-channel stimulus file.")
            continue

        # Transpose to (samples, channels)
        audio_array = audio_array.T

        # Save as .wav
        sf.write(wav_path, audio_array, samplerate=44100)
        print(f"  → Saved: {wav_path}\n")
