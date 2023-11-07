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


flattened_matrices = [matrix.values[np.triu_indices(76)] for matrix in combined_data_st['st_matrix']]
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
covars = {'batch':batch_vector,
          'gender':sex_vector}
covars = pd.DataFrame(covars)

# To specify names of the variables that are categorical:
categorical_cols = ['gender']

# To specify the name of the variable that encodes for the scanner/batch covariate:
batch_col = 'batch'

#Harmonization step:
data_combat = neuroCombat(dat=data,
    covars=covars,
    batch_col=batch_col,
    categorical_cols=categorical_cols)["data"]

print(data.shape)
print(data_combat.shape)