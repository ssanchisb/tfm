import pandas as pd
import numpy as np
import os.path

patient_info = pd.read_csv('/home/vant/code/tfm1/data/clinic.csv', usecols=['id', 'age', 'sex'])

# we are normalizing between 0 and 1 the files that have been previously harmonized with neuroCombat
path_st = '/home/vant/code/tfm1/data/structural_h'
path_func = '/home/vant/code/tfm1/data/functional_h'

csv_files_st = [file for file in sorted(os.listdir(path_st))]
csv_files_func = [file for file in sorted(os.listdir(path_func))]

st_matrices = [pd.read_csv(os.path.join(path_st, file), header=None) for file in csv_files_st]
func_matrices = [pd.read_csv(os.path.join(path_func, file), header=None) for file in csv_files_func]

# Find the global maximum value across all matrices
global_max_st = max(df.values.max() for df in st_matrices)
global_max_func = max(df.values.max() for df in func_matrices)

# Normalize each matrix based on the global maximum
structural_norm = [(df / global_max_st) for df in st_matrices]
functional_norm = [(df / global_max_st) for df in func_matrices]
"""
print(global_max_func)
print(func_matrices[0])
print(functional_norm[0])
"""
output_dir_st = '/home/vant/code/tfm1/data/structural_norm'
os.makedirs(output_dir_st, exist_ok=True)

output_dir_func = '/home/vant/code/tfm1/data/functional_norm'
os.makedirs(output_dir_func, exist_ok=True)

for matrix, patient_id in zip(structural_norm, patient_info['id']):
    # Define the filename using the 'id' field
    filename = os.path.join(output_dir_st, f'{patient_id}.csv')
    # Save the matrix as a CSV file
    matrix.to_csv(filename, index=False, header=None)

for matrix, patient_id in zip(functional_norm, patient_info['id']):
    # Define the filename using the 'id' field
    filename = os.path.join(output_dir_func, f'{patient_id}.csv')
    # Save the matrix as a CSV file
    matrix.to_csv(filename, index=False, header=None)

