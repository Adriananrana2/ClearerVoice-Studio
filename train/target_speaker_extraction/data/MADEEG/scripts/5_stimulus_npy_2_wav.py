import numpy as np
import soundfile as sf
import os

# Input directory containing all .npy stimulus files
input_dir = "../preprocessed_npy/stimulus_npy/"

# Output directory to save all .wav files
output_dir = "../processed_data/stimulus_wav/"
os.makedirs(output_dir, exist_ok=True)  # Create it if it doesn't exist

# Loop through all .npy files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".npy"):
        npy_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        wav_path = os.path.join(output_dir, base_name + ".wav")

        # Load the numpy array
        audio_array = np.load(npy_path)
        print(f"Processing {filename} ... Shape: {audio_array.shape}")

        # Check if it's a valid stereo file (2 channels)
        if audio_array.shape[0] != 2:
            print("  → Skipped: Not a 2-channel stimulus file.")
            continue

        # Transpose to (samples, channels)
        audio_array = audio_array.T

        # Save as .wav to the output directory
        sf.write(wav_path, audio_array, samplerate=44100)
        print(f"  → Saved: {wav_path}\n")
