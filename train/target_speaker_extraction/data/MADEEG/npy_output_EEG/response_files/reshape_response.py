import os
import numpy as np

# Define directories
input_dir = './'  # current directory (where this script is located)
output_dir = '../response_files_reshaped'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Desired shape
target_shape = (20, 5052)

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
        
        # Cut to shape (20, 5052)
        reshaped_data = data[:, :5052]
        
        # Save to new directory
        output_path = os.path.join(output_dir, file_name)
        np.save(output_path, reshaped_data)
        print(f"Saved reshaped file: {output_path}")
