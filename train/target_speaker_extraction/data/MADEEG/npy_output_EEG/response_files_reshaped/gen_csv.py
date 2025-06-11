import os
import csv

# Define working directory
folder_path = './'
output_csv = os.path.join(folder_path, 'file_list.csv')

# Collect rows
rows = []

# Loop over .npy files
for file_name in os.listdir(folder_path):
    if file_name.endswith('.npy'):
        # Remove file extension
        name_without_ext = file_name[:-4]  # remove '.npy'
        
        # Split at first underscore
        parts = name_without_ext.split('_', 1)
        if len(parts) == 2:
            first_col = parts[0]
            second_col = parts[1]
            rows.append([first_col, second_col])

# Write to CSV (no header)
with open(output_csv, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(rows)

print(f"CSV file saved to: {output_csv}")
