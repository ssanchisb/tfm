import pandas as pd
import os.path
import numpy as np
from neuroCombat import neuroCombat


patient_info = pd.read_csv('/home/vant/code/tfm1/data/clinic.csv', usecols=['id', 'age', 'sex'])

path_func = '/home/vant/code/tfm1/data/functional'

csv_files_func = [file for file in sorted(os.listdir(path_func))]

func_matrices = [pd.read_csv(os.path.join(path_func, file), header=None) for file in csv_files_func]

np_matrices = [m.values.reshape(76,76) for m in func_matrices]
for m in np_matrices:
    m[np.diag_indices_from(m)] = 0.0

zero_matrices = [pd.DataFrame(m) for m in np_matrices]

"""
try_diag = pd.DataFrame(func_matrices[0])
n_diag = try_diag.values.reshape((76,76))
n_diag[np.diag_indices_from(try_diag)] = 0

"""
print(type(zero_matrices[0]))