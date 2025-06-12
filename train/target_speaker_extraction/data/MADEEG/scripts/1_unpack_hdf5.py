import os
import h5py
import shutil
import numpy as np

hdf5_file = '../madeeg_preprocessed.hdf5'
output_dir = '.././preprocessed_npy/'  
os.makedirs(output_dir, exist_ok=True)

def save_datasets_to_npy(hdf5_obj, current_path=""):
    """Recursively save all datasets in an HDF5 file to .npy files."""
    for key in hdf5_obj.keys():
        item = hdf5_obj[key]
        path = f"{current_path}/{key}" if current_path else key

        if isinstance(item, h5py.Dataset):
            
            data = item[:]
            output_file = os.path.join(output_dir, path.replace('/', '_') + '.npy')
            print(f"Saving: {path} → {output_file} (Shape: {data.shape}, Dtype: {data.dtype})")
            np.save(output_file, data)
        elif isinstance(item, h5py.Group):
            
            save_datasets_to_npy(item, path)

with h5py.File(hdf5_file, 'r') as f:
    save_datasets_to_npy(f)

print("All datasets saved as .npy files!")

response_folder = os.path.join(output_dir, 'responses_npy')
soli_folder = os.path.join(output_dir, 'soli_npy')
stimulus_folder = os.path.join(output_dir, 'stimulus_npy')
os.makedirs(response_folder, exist_ok=True)
os.makedirs(soli_folder, exist_ok=True)
os.makedirs(stimulus_folder, exist_ok=True)

for filename in os.listdir(output_dir):
    if filename.endswith('.npy'):
        source_path = os.path.join(output_dir, filename)
     
        if '_response' in filename:
            shutil.move(source_path, os.path.join(response_folder, filename))
        elif '_soli' in filename:
            shutil.move(source_path, os.path.join(soli_folder, filename))
        elif '_stimulus' in filename:
            shutil.move(source_path, os.path.join(stimulus_folder, filename))

print("NPYs ARCHIVED！")