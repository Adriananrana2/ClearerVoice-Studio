import os
import numpy as np
import soundfile as sf

# Define input/output directories
input_dir = "./"
output_dir = "../isolated_songs/"

# Track processed songs to avoid duplicates
seen_songs = set()

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

    # Load audio and extract solos
    audio_array = np.load(npy_path)  # shape: (3, N)
    num_instruments = len(instruments)

    for i in range(num_instruments):
        solo_audio = audio_array[i]
        solo_name = f"{song_id}_{instruments[i]}_soli.wav"
        solo_path = os.path.join(output_dir, solo_name)
        sf.write(solo_path, solo_audio, samplerate=44100)
        print(f"Saved: {solo_path}")
