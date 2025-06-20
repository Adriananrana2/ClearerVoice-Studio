import os
import numpy as np
import soundfile as sf

# Define input/output directories
input_dir = "../preprocessed_npy/soli_npy/"
output_dir = "../processed_data/isolated_wav/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Track processed songs to avoid duplicates
seen_songs = set()

# Set fixed target length: 19 seconds at 44100 Hz
TARGET_LENGTH = 44100 * 19  # 837900 samples

# Loop through all .npy files
for fname in os.listdir(input_dir):
    if not fname.endswith(".npy"):
        continue

    parts = fname.split("_")
    song_id = "_".join(parts[0:7])  # Unique song identity, e.g. "0001_classique_morceau1_duo_CoFl_theme1_stereo"

    if song_id in seen_songs:
        continue  # Skip duplicate entries

    seen_songs.add(song_id)

    npy_path = os.path.join(input_dir, fname)
    print(f"Processing: {npy_path}")

    # Extract instrument code, e.g. "CoFl" → ["Co", "Fl"]
    inst_code = parts[4]
    instruments = [inst_code[i:i+2] for i in range(0, len(inst_code), 2)]

    # Load audio and isolate
    audio_array = np.load(npy_path)  # shape: (3, N)
    num_instruments = len(instruments)

    for i in range(num_instruments):
        isolated_audio = audio_array[i]
        isolated_audio = isolated_audio[:TARGET_LENGTH] # Trim to 19 seconds max
        isolated_name = f"{song_id}_{instruments[i]}_soli.wav"
        isolated_path = os.path.join(output_dir, isolated_name)
        sf.write(isolated_path, isolated_audio, samplerate=44100)
        print(f"Saved: {isolated_path}")
