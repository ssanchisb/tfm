import pandas as pd
import os.path
import numpy as np
from neuroCombat import neuroCombat


patient_info = pd.read_csv('/home/vant/code/tfm1/data/clinic.csv', usecols=['id', 'age', 'sex'])

path_func = '/home/vant/code/tfm1/data/functional'

csv_files_func = [file for file in sorted(os.listdir(path_func))]

func_matrices = [pd.read_csv(os.path.join(path_func, file), header=None) for file in csv_files_func]

# Functional matrices have weight=1 on all diagonals
# We first convert all diagonal values to zero
np_matrices = [m.values.reshape(76,76) for m in func_matrices]

zero_matrices = [pd.DataFrame(m.copy()) for m in np_matrices]

for m in zero_matrices:
    m.values[np.diag_indices_from(m)] = 0.0
    m.values[m.values < 0] = 0

zero_matrices = [pd.DataFrame(m) for m in zero_matrices]


print(type(zero_matrices[0]))
print(len(zero_matrices))
print(func_matrices[0])
print(zero_matrices[0])

