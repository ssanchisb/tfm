import pandas as pd
import os
import shutil

source_structural = '/home/vant/code/tfm1/data/subject_networks_FA_v1'
destination_structural = '/home/vant/code/tfm1/data/structural'
source_functional = '/home/vant/code/tfm1/data/subject_networks_rfMRI_v1'
destination_functional = '/home/vant/code/tfm1/data/functional'

path1 = '/home/vant/code/tfm1/data/subject_networks_FA_v1'
path2 = '/home/vant/code/tfm1/data/subject_networks_rfMRI_v1'

# Load the patient information data
patient_info = pd.read_csv('/home/vant/code/tfm1/data/clinic.csv', usecols=['id', 'age', 'sex'])

# Get the list of patient IDs
patient_ids = patient_info['id'].astype(str)

# Find CSV files in the source folders
csv_files_st = [file for file in os.listdir(path1) if file.endswith('.csv')]
csv_files_func = [file for file in os.listdir(path2) if file.endswith('.csv')]

# Filter CSV files based on patient IDs and short names
filtered_csv_files_st = [file for file in csv_files_st if file.replace('_FA_factor.csv', '') in patient_ids.values]
filtered_csv_files_func = [file for file in csv_files_func if file.replace('_r_matrix.csv', '') in patient_ids.values]

# Create a new folder for the filtered CSV files
os.makedirs(destination_structural, exist_ok=True)
os.makedirs(destination_functional, exist_ok=True)

# Copy the filtered CSV files to the new folders with short names
for file in filtered_csv_files_st:
    new_name = file.replace('_FA_factor.csv', '') + '.csv'  # Short name
    old_path_st = os.path.join(source_structural, file)
    new_path_st = os.path.join(destination_structural, new_name)
    shutil.copy(old_path_st, new_path_st)

for file in filtered_csv_files_func:
    new_name = file.replace('_r_matrix.csv', '') + '.csv'  # Short name
    old_path_func = os.path.join(source_functional, file)
    new_path_func = os.path.join(destination_functional, new_name)
    shutil.copy(old_path_func, new_path_func)


items = os.listdir('/home/vant/code/tfm1/data/structural')
print(len(items))