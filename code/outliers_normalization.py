import pandas as pd
import os.path
from scipy.stats import iqr
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

patient_info = pd.read_csv('/home/vant/code/tfm1/data/clinic.csv', usecols=['id', 'age', 'sex'])

path_st = '/home/vant/code/tfm1/data/structural_h'
path_func = '/home/vant/code/tfm1/data/functional_h'

csv_files_st = [file for file in sorted(os.listdir(path_st), key=lambda x: x.lower())]
csv_files_func = [file for file in sorted(os.listdir(path_func), key=lambda x: x.lower())]

st_matrices = [pd.read_csv(os.path.join(path_st, file), header=None) for file in csv_files_st]
func_matrices = [pd.read_csv(os.path.join(path_func, file), header=None) for file in csv_files_func]


fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Boxplot of Structural Matrices
axes[0].boxplot([df.values.flatten() for df in st_matrices])
axes[0].set_title('Boxplot of Structural Matrices')

# Boxplot of Functional Matrices
axes[1].boxplot([df.values.flatten() for df in func_matrices])
axes[1].set_title('Boxplot of Functional Matrices')

# Adjust layout and show the plots
plt.tight_layout()
plt.show()


matrix_values_flattened_st = np.concatenate([df.values.flatten() for df in st_matrices])

outliers_st = []
for df in st_matrices:
    for value in df.values.flatten():
        if value > 1.1:
            outliers_st.append(value)

print(outliers_st)
print(len(outliers_st))

percentile_to_exclude_st = 99.99
max_value_st = np.percentile(matrix_values_flattened_st, percentile_to_exclude_st)
print("max structural value: {}".format(max(matrix_values_flattened_st)))
print("Upper threshold value structural: {}".format(max_value_st))

normalized_st_matrices = [((df - df.min().min()) / (max_value_st - df.min().min())).clip(upper=1) for df in st_matrices]

# Print the outliers and their count
outliers_st = [value for value in matrix_values_flattened_st if value > max_value_st]

print("Number of outliers structural:", len(outliers_st))

print(np.max(matrix_values_flattened_st))
print(np.max(normalized_st_matrices))

#functional matrices normalization

matrix_values_flattened_func = np.concatenate([df.values.flatten() for df in func_matrices])

max_func = np.max(matrix_values_flattened_func)
print("max func: {}".format(max_func))

percentile_to_exclude_func = 99.99
max_value_func = np.percentile(matrix_values_flattened_func, percentile_to_exclude_func)
print("Upper threshold value func: {}".format(max_value_func))

normalized_func_matrices = [((df - df.min().min()) / (max_value_func - df.min().min())).clip(upper=1) for df in func_matrices]
normalized_func_flattened = np.concatenate([df.values.flatten() for df in normalized_func_matrices])

plt.figure(figsize=(10, 6))
sns.histplot(matrix_values_flattened_func, bins=20, kde=True)
plt.title('Histogram of Functional Matrix Values')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

# Print the outliers and their count
outliers_func = [value for value in matrix_values_flattened_func if value > max_value_func]
print("Number of outliers func:", len(outliers_func))
print(np.max(normalized_func_flattened))


# Export to .csv files
output_dir_st = '/home/vant/code/tfm1/data/structural_norm'
os.makedirs(output_dir_st, exist_ok=True)

output_dir_func = '/home/vant/code/tfm1/data/functional_norm'
os.makedirs(output_dir_func, exist_ok=True)

for matrix, patient_id in zip(normalized_st_matrices, patient_info['id']):
    # Define the filename using the 'id' field
    filename = os.path.join(output_dir_st, f'{patient_id}.csv')
    # Save the matrix as a CSV file
    matrix.to_csv(filename, index=False, header=None)

for matrix, patient_id in zip(normalized_func_matrices, patient_info['id']):
    # Define the filename using the 'id' field
    filename = os.path.join(output_dir_func, f'{patient_id}.csv')
    # Save the matrix as a CSV file
    matrix.to_csv(filename, index=False, header=None)