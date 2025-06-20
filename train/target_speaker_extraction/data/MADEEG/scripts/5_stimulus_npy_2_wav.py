import numpy as np
import soundfile as sf
import os

# Input and output directories
input_dir = "../preprocessed_npy/stimulus_npy/"
output_dir = "../processed_data/stimulus_wav/"
os.makedirs(output_dir, exist_ok=True)

# Fixed duration
TARGET_LENGTH = 44100 * 19  # 837900 samples

# Loop through all .npy files
for filename in os.listdir(input_dir):
    if filename.endswith(".npy"):
        npy_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        wav_path = os.path.join(output_dir, base_name + ".wav")

        # Load array
        audio_array = np.load(npy_path)
        print(f"Processing {filename} ... Shape: {audio_array.shape}")

        # Ensure 2-channel (stereo)
        if audio_array.shape[0] != 2:
            print("  → Skipped: Not a 2-channel stimulus file.")
            continue

        # Trim to 19 seconds
        audio_array = audio_array[:, :TARGET_LENGTH]

        # Transpose for saving: (samples, channels)
        audio_array = audio_array.T

        # Save to wav
        sf.write(wav_path, audio_array, samplerate=44100)
        print(f"  → Saved: {wav_path}\n")
