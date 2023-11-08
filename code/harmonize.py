import pandas as pd
import os.path
import numpy as np
from neuroCombat import neuroCombat


patient_info = pd.read_csv('/home/vant/code/tfm1/data/clinic.csv', usecols=['id', 'age', 'sex'])

path_st = '/home/vant/code/tfm1/data/structural'
path_func = '/home/vant/code/tfm1/data/functional'

csv_files_st = [file for file in sorted(os.listdir(path_st))]
csv_files_func = [file for file in sorted(os.listdir(path_func))]

st_matrices = [pd.read_csv(os.path.join(path_st, file), header=None) for file in csv_files_st]
func_matrices = [pd.read_csv(os.path.join(path_func, file), header=None) for file in csv_files_func]

# from here on I continue with structural matrices only

combined_data_st = patient_info.copy()
combined_data_st['st_matrix'] = st_matrices


flattened_matrices = [matrix.values[np.triu_indices(76)] for matrix in st_matrices]
flattened_data = pd.DataFrame(flattened_matrices)
concatenated_patients = flattened_data.T

#print(concatenated_patients)
#print(patient_info)

batch_vector = [1 if 'MSVIS' in id else 2 for id in patient_info['id']]
sex_vector = patient_info['sex'].replace({0: 1, 1: 2}).tolist()

"""
print(batch_vector)
print(sex_vector)

if len(batch_vector) == len(sex_vector) == len(patient_info):
    print("Vectors have the same length as patient_info.")
else:
    print("Vectors do not have the same length as patient_info.")

"""


# Getting example data
# 200 rows (features) and 10 columns (scans)
data = concatenated_patients


# Specifying the batch (scanner variable) as well as a biological covariate to preserve:
covars = {'batch':batch_vector}
covars = pd.DataFrame(covars)

# To specify names of the variables that are categorical:
categorical_cols = []

# To specify the name of the variable that encodes for the scanner/batch covariate:
batch_col = 'batch'

#Harmonization step:
data_combat = neuroCombat(dat=data, covars=covars, batch_col=batch_col, eb=False)["data"]

print(data.shape)
print(data_combat.shape)



num_matrices = data_combat.shape[1]
matrix_size = 76

# Initialize an empty list to store the reshaped DataFrames
reshaped_matrices = []

# Iterate through each flattened matrix in data_combat
for i in range(num_matrices):
    flattened_matrix = data_combat[:, i]

    # Create an empty matrix filled with zeros
    symmetric_matrix = np.zeros((matrix_size, matrix_size))

    # Calculate the indices for the upper triangle of the matrix
    triu_indices = np.triu_indices(matrix_size)

    # Fill the upper triangle with values from the flattened array
    symmetric_matrix[triu_indices] = flattened_matrix

    # Fill the lower triangle to make the matrix symmetric
    symmetric_matrix = symmetric_matrix + symmetric_matrix.T - np.diag(symmetric_matrix.diagonal())

    # Convert the symmetric matrix to a DataFrame
    matrix_df = pd.DataFrame(symmetric_matrix)

    # Append the DataFrame to the list
    reshaped_matrices.append(matrix_df)

# Now, reshaped_matrices is a list of 165 DataFrames representing symmetric matrices with a shape of (76, 76)

# Iterate through the matrices in reshaped_matrices and the corresponding matrices in st_matrices
for i in range(len(reshaped_matrices)):
    # Find the positions where the original_matrix has values equal to 0
    zero_positions = st_matrices[i] == 0

    # Set the corresponding positions in the reshaped_matrix to 0
    reshaped_matrices[i].values[zero_positions] = 0

# Now, reshaped_matrices has the values set to 0 only at the same positions where st_matrices have 0



print(st_matrices[1])
print(reshaped_matrices[1])


"""
# Initialize a flag to check if there are negative values
has_negative_values = False

# Iterate through the matrices in reshaped_matrices
for matrix in reshaped_matrices:
    if (matrix.values < 0).any():
        has_negative_values = True
        break  # Exit the loop as soon as a negative value is found

# Check if there are any negative values in the reshaped_matrices
if has_negative_values:
    print("There are negative values in reshaped_matrices.")
else:
    print("There are no negative values in reshaped_matrices.")

"""


# Create a directory to store the CSV files
output_dir = '/home/vant/code/tfm1/data/structural_h'  # Change this to your desired directory
os.makedirs(output_dir, exist_ok=True)

# Iterate through reshaped_matrices and patient_info to save each matrix with the corresponding 'id' as the filename
for matrix, patient_id in zip(reshaped_matrices, patient_info['id']):
    # Define the filename using the 'id' field
    filename = os.path.join(output_dir, f'{patient_id}.csv')

    # Save the matrix as a CSV file
    matrix.to_csv(filename, index=False, header=None)

#merging try

