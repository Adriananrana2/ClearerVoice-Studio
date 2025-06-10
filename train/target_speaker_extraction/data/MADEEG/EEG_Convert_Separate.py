import os
import shutil

source_folder = './npy_output_EEG' 
response_folder = os.path.join(source_folder, 'response_files')
soli_folder = os.path.join(source_folder, 'soli_files')
stimulus_folder = os.path.join(source_folder, 'stimulus_files')
os.makedirs(response_folder, exist_ok=True)
os.makedirs(soli_folder, exist_ok=True)
os.makedirs(stimulus_folder, exist_ok=True)

for filename in os.listdir(source_folder):
    if filename.endswith('.npy'):
        source_path = os.path.join(source_folder, filename)
     
        if '_response' in filename:
            shutil.move(source_path, os.path.join(response_folder, filename))
        elif '_soli' in filename:
            shutil.move(source_path, os.path.join(soli_folder, filename))
        elif '_stimulus' in filename:
            shutil.move(source_path, os.path.join(stimulus_folder, filename))

print("DOWN！")