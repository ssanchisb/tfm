import pandas as pd
import os.path
import shutil

source_structural = '/home/vant/code/tfm1/data/subject_networks_FA_v1'
destination_structural = '/home/vant/code/tfm1/data/structural'
source_functional = '/home/vant/code/tfm1/data/subject_networks_rfMRI_v1'
destination_functional = '/home/vant/code/tfm1/data/functional'

path1 = '/home/vant/code/tfm1/data/subject_networks_FA_v1'
path2 = '/home/vant/code/tfm1/data/subject_networks_rfMRI_v1'

csv_files_st = [file for file in os.listdir(path1) if file.endswith('.csv')]
csv_files_func = [file for file in os.listdir(path2) if file.endswith('.csv')]

# Rename the files by removing the substring "_FA_factor" from the filename
for file in csv_files_st:
    new_name = file.replace('_FA_factor', '')  # Create the new name
    old_path_st = os.path.join(source_structural, file)  # Full path to the original file
    new_path_st = os.path.join(destination_structural, new_name)  # Full path to the new file name

    shutil.copy(old_path_st, new_path_st)  # Rename the file

for file in csv_files_func:
    new_name = file.replace('_r_matrix', '')
    old_path_func = os.path.join(source_functional, file)
    new_path_func = os.path.join(destination_functional, new_name)

    shutil.copy(old_path_func, new_path_func)


path_st = '/home/vant/code/tfm1/data/structural'
path_func = '/home/vant/code/tfm1/data/functional'

csv_files_st = [file for file in sorted(os.listdir(path_st))]
csv_files_func = [file for file in sorted(os.listdir(path_func))]

st_matrices = [pd.read_csv(os.path.join(path_st, file), header=None) for file in csv_files_st]
func_matrices = [pd.read_csv(os.path.join(path_func, file), header=None) for file in csv_files_func]

patient_info = pd.read_csv('/home/vant/code/tfm1/data/clinic.csv', usecols=['id', 'age', 'sex'])

patient_ids = patient_info['id']

filtered_st_matrices = [st for st, file in zip(st_matrices, csv_files_st) if file.replace('.csv', '') in patient_ids.values]
filtered_func_matrices = [st for st, file in zip(func_matrices, csv_files_func) if file.replace('.csv', '') in patient_ids.values]

#print(patient_ids)
print(len(patient_info))
print(len(filtered_st_matrices))
#print(filtered_st_matrices[0])
print(len(filtered_func_matrices))
print(filtered_func_matrices[0])




