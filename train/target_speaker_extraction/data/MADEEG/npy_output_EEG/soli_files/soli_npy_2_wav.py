import numpy as np
import soundfile as sf
import os

# Directory containing all .npy files
soli_dir = "./"

# Loop through all .npy files in the directory
for filename in os.listdir(soli_dir):
    if filename.endswith(".npy"):
        npy_path = os.path.join(soli_dir, filename)
        base_name = os.path.splitext(filename)[0]
        wav_path = os.path.join(soli_dir, base_name + ".wav")

        # Load the numpy array
        audio_array = np.load(npy_path)

        print(f"Processing {filename} ... Shape: {audio_array.shape}")

        # Handle DUO case by removing the 3rd silent channel
        if audio_array.shape[0] == 3 and np.allclose(audio_array[2], 0):
            print("  → Detected DUO. Using first 2 channels.")
            audio_array = audio_array[:2, :]
        elif audio_array.shape[0] == 3:
            print("  → Detected TRIO. Using all 3 channels.")
        else:
            print("  → Unexpected shape. Skipping.")
            continue

        # Transpose to (samples, channels)
        audio_array = audio_array.T

        # Save as .wav
        sf.write(wav_path, audio_array, samplerate=44100)
        print(f"  → Saved: {wav_path}\n")
