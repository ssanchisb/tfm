import copy
import os.path
import pandas as pd
import numpy as np

# This script filters the edges of the structural matrices
# by removing all values <0.1 and then removing all edges
# that are not present in more than 60% of the controls.
# The same edges are removed from the functional matrices.
# The resulting matrices are saved in the folder 'data/structural_ready'
# and 'data/functional_ready' respectively.


# extract labels of MS vs. HV:
demographics_df = pd.read_csv('/home/vant/code/tfm1/data/clinic.csv')
labels = demographics_df['controls_ms'].tolist()

# obtain matrices from normalized data:
path_st = '/home/vant/code/tfm1/data/structural_norm'
path_func = '/home/vant/code/tfm1/data/functional_norm'

csv_files_st = [file for file in sorted(os.listdir(path_st), key=lambda x: x.lower())]
csv_files_func = [file for file in sorted(os.listdir(path_func), key=lambda x: x.lower())]

st_matrices = [pd.read_csv(os.path.join(path_st, file), header=None) for file in csv_files_st]
func_matrices = [pd.read_csv(os.path.join(path_func, file), header=None) for file in csv_files_func]

# Remove values <0.1 in structural matrices
threshold_st = []

for matrix in st_matrices:
    modified_matrix = matrix.copy()
    modified_matrix[modified_matrix < 0.1] = 0
    threshold_st.append(modified_matrix)

# use labels to separate structural matrices into 2 groups:
ms_st = [threshold_st[i] for i, value in enumerate(labels) if value == 1]
hv_st = [threshold_st[i] for i, value in enumerate(labels) if value == 0]

# Flatten upper triangles for further processing
flattened_matrices = [matrix.values[np.triu_indices(76)] for matrix in hv_st]
flattened_data = pd.DataFrame(flattened_matrices)
concatenated_controls = flattened_data

# turn to 0 all values that are not present in more than 60% of controls:
filtered_hv = concatenated_controls.copy()
num_columns_turned_to_zero = 0
for column in filtered_hv:
    if (filtered_hv[column] == 0).mean() > 0.4:
        filtered_hv[column] = 0
        num_columns_turned_to_zero += 1

num_columns_non_zero = (filtered_hv != 0).any().sum()
print(f"Number of nodes turned to zero after 60% filtering: {num_columns_turned_to_zero}")
print(f"Number of nodes with non-zero values after filtering: {num_columns_non_zero}")

# reshape the matrices from flattened arrays:
flattened_hv60 = filtered_hv.values

num_matrices = len(flattened_hv60)
matrix_size = 76

# Initialize a list to store the unflattened matrices as DataFrames
unflattened_hv60 = []

# Use np.triu_indices to get the upper triangular indices
upper_tri_indices = np.triu_indices(matrix_size)

# Fill the upper triangular part of each matrix and create a DataFrame
for i in range(num_matrices):
    matrix_data = np.zeros((matrix_size, matrix_size))
    matrix_data[upper_tri_indices[0], upper_tri_indices[1]] = flattened_hv60[i]

    # Fill the lower triangular part by copying the transposed upper triangular part
    matrix_data.T[upper_tri_indices[0], upper_tri_indices[1]] = flattened_hv60[i]

    matrix_df = pd.DataFrame(matrix_data)
    unflattened_hv60.append(matrix_df)

# Find the indices where all hv control matrices have 0 values
avg_hv60 = pd.concat(unflattened_hv60).groupby(level=0).mean()

zero_positions = avg_hv60 == 0

# Create new matrices where mask with 0 values on the same
# indices where more than 60% of hv had no edge weight
structural_threshold = copy.deepcopy(st_matrices)
functional_threshold = copy.deepcopy(func_matrices)

for matrix in structural_threshold:
    matrix[zero_positions] = 0
for matrix in functional_threshold:
    matrix[zero_positions] = 0

# Export the masked matrices as .csv files
output_dir_st = '/home/vant/code/tfm1/data/structural_ready'
os.makedirs(output_dir_st, exist_ok=True)

output_dir_func = '/home/vant/code/tfm1/data/functional_ready'
os.makedirs(output_dir_func, exist_ok=True)

for matrix, patient_id in zip(structural_threshold, demographics_df['id']):
    # Define the filename using the 'id' field
    filename = os.path.join(output_dir_st, f'{patient_id}.csv')
    # Save the matrix as a CSV file
    matrix.to_csv(filename, index=False, header=None)

for matrix, patient_id in zip(functional_threshold, demographics_df['id']):
    # Define the filename using the 'id' field
    filename = os.path.join(output_dir_func, f'{patient_id}.csv')
    # Save the matrix as a CSV file
    matrix.to_csv(filename, index=False, header=None)
