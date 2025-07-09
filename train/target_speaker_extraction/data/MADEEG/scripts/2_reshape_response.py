import os
import numpy as np
from scipy.signal import resample_poly  # Upsampling

# Define directories
input_dir = '../preprocessed_npy/responses_npy/'  # current directory (where this script is located)
output_dir = '../processed_data/response_npy/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Desired shape
upsample_factor = 5  # Upsample EEG from 256Hz → 1280Hz

# Loop through .npy files
for file_name in os.listdir(input_dir):
    if file_name.endswith('.npy'):
        file_path = os.path.join(input_dir, file_name)
        
        # Load the response file
        data = np.load(file_path)
        
        # Check shape
        if data.shape[0] != 20:
            print(f"Skipping {file_name}: unexpected number of channels {data.shape[0]}")
            continue
        
        # Cut to shape (20, 4864)
        reshaped_data = data[:, :4864]

        # ✅ Upsample to (20, 4864 * 5) = (20, 24320)
        upsampled_data = resample_poly(reshaped_data, up=upsample_factor, down=1, axis=-1)

        # Save to new directory
        output_path = os.path.join(output_dir, file_name)
        np.save(output_path, upsampled_data)
        print(f"Saved upsampled file: {output_path} (shape: {upsampled_data.shape})")
