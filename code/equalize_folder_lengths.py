import pandas as pd
import os
import shutil

# This script is used to equalize the length of the folders containing the structural and functional data.
# The structural and functional data are stored in two folders, one for each type of data. The names of the files
# in these folders are the patient IDs. The patient IDs are also stored in a CSV file called clinic.csv. This file
# contains the patient IDs and some other information about the patients. The structural and functional data contains
# some extra information in the file names, so the first step is to remove this extra information from the file names
# and keep only the patient IDs. Then, the patient IDs are compared with the ones in clinic.csv. If a patient ID is
# present in both clinic.csv and the structural/functional data, then the corresponding CSV file is copied to a new
# folder. The files in the new folder have the same name as the patient IDs. Finally, the data from volum_nodes_patients
# and volum_nodes_controls is exported to CSV files.

# The folder containing the structural data is called subject_networks_FA_v1 and the folder containing the functional
# data is called subject_networks_rfMRI_v1. The structural data is stored in CSV files with names like
# 'r100307_FA_factor.csv'. The functional data is stored in CSV files with names like 'r100307_r_matrix.csv'. The
# patient IDs are numbers, so the first step is to remove the extra information from the file names and keep only the
# patient IDs. Then, the patient IDs are compared with the ones in clinic.csv. If a patient ID is present in both
# clinic.csv and the structural/functional data, then the corresponding CSV file is copied to a new folder. The files
# in the new folder have the same name as the patient IDs. Finally, the data from volum_nodes_patients and
# volum_nodes_controls is exported to CSV files.


# Define source files:
source_structural = '/home/vant/code/tfm1/data/subject_networks_FA_v1'
destination_structural = '/home/vant/code/tfm1/data/structural'
source_functional = '/home/vant/code/tfm1/data/subject_networks_rfMRI_v1'
destination_functional = '/home/vant/code/tfm1/data/functional'

# Load the patient information data
patient_info = pd.read_csv('/home/vant/code/tfm1/data/clinic.csv', usecols=['id', 'age', 'sex'])

# Load data from VOLUM_NODES_CONTROLS.xls and VOLUM_NODES_PATIENTS.xls
volum_nodes_controls = pd.read_excel('/home/vant/code/tfm1/data/VOLUM_NODES_CONTROLS.xls', header=None, skiprows=1)
volum_nodes_patients = pd.read_excel('/home/vant/code/tfm1/data/VOLUM_NODES_PATIENTS.xls', header=None, skiprows=1)

# Get the list of patient IDs
patient_ids = patient_info['id'].astype(str)

# Find CSV files in the source folders
csv_files_st = [file for file in os.listdir(source_structural) if file.endswith('.csv')]
csv_files_func = [file for file in os.listdir(source_functional) if file.endswith('.csv')]

# Remove extra part of the file names and include only those present in clinic.csv file
filtered_csv_files_st = [file for file in csv_files_st if file.replace('_FA_factor.csv', '') in patient_ids.values]
filtered_csv_files_func = [file for file in csv_files_func if file.replace('_r_matrix.csv', '') in patient_ids.values]

# Remove starting 'r' in all values of the first column of volum_nodes_patients and volum_nodes_controls
volum_nodes_patients[0] = volum_nodes_patients[0].str.replace('r', '')
volum_nodes_controls[0] = volum_nodes_controls[0].str.replace('r', '')

# Remove rows from volum_nodes_patients where the first column value is not in patient_ids and redo the index
volum_nodes_patients = volum_nodes_patients[volum_nodes_patients[0].isin(patient_ids.values)].reset_index(drop=True)

# Export the dataframes as csv files to a folder
volum_nodes_patients.to_csv('/home/vant/code/tfm1/data/volum_nodes_patients.csv', index=False)
volum_nodes_controls.to_csv('/home/vant/code/tfm1/data/volum_nodes_controls.csv', index=False)

# Create a new folder for the filtered CSV files
os.makedirs(destination_structural, exist_ok=True)
os.makedirs(destination_functional, exist_ok=True)

# Copy the filtered CSV files to the new folders with short names
for file in filtered_csv_files_st:
    new_name = file.replace('_FA_factor.csv', '') + '.csv'
    old_path_st = os.path.join(source_structural, file)
    new_path_st = os.path.join(destination_structural, new_name)
    shutil.copy(old_path_st, new_path_st)

for file in filtered_csv_files_func:
    new_name = file.replace('_r_matrix.csv', '') + '.csv'
    old_path_func = os.path.join(source_functional, file)
    new_path_func = os.path.join(destination_functional, new_name)
    shutil.copy(old_path_func, new_path_func)

# Check for equal amount of files in all folders
items_st = os.listdir('/home/vant/code/tfm1/data/structural')
items_func = os.listdir('/home/vant/code/tfm1/data/functional')
print(len(items_st) == len(items_func) == len(patient_ids))
