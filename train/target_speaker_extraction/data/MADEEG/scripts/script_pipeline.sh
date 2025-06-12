#!/bin/bash

# Navigate to the directory of this script
cd "$(dirname "$0")"

echo "Starting pipeline execution..."

echo "1. Unpacking HDF5..."
python 1_unpack_hdf5.py || { echo "❌ Failed at step 1_unpack_hdf5.py"; exit 1; }

echo "2. Reshaping response..."
python 2_reshape_response.py || { echo "❌ Failed at step 2_reshape_response.py"; exit 1; }

echo "3. Generating CSV..."
python 3_gen_csv.py || { echo "❌ Failed at step 3_gen_csv.py"; exit 1; }

echo "4. Isolating mixed songs..."
python 4_isolate_mixed_songs.py || { echo "❌ Failed at step 4_isolate_mixed_songs.py"; exit 1; }

echo "5. Converting stimulus .npy to .wav..."
python 5_stimulus_npy_2_wav.py || { echo "❌ Failed at step 5_stimulus_npy_2_wav.py"; exit 1; }

echo "✅ All scripts executed successfully!"
