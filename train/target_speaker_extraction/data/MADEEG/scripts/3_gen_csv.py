import os
import csv
import random

# Set random seed for reproducibility
random.seed(42)

# Define working directory
input_dir = '../processed_data/response_npy/'
output_dir = '../processed_data/'
output_csv = os.path.join(output_dir, 'file_list.csv')

# Collect all .npy filenames (no suffix)
file_names = [
    fname[:-4]  # remove '.npy'
    for fname in os.listdir(input_dir)
    if fname.endswith('.npy')
]

# Shuffle the list
random.shuffle(file_names)

# Split
train_files = file_names[:200]
test_files = file_names[200:223]
val_files = file_names[223:]

# Build rows with dataset label
rows = []
for name in train_files:
    rows.append(['train', name])
for name in test_files:
    rows.append(['test', name])
for name in val_files:
    rows.append(['val', name])

# Save to CSV (no header)
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"CSV file saved to: {output_csv}")
