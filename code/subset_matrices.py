import pandas as pd
import os.path


patient_info = pd.read_csv('/home/vant/code/tfm1/data/clinic.csv', usecols=['id', 'age', 'sex'])
controls = pd.read_csv('/home/vant/code/tfm1/data/clinic.csv', usecols=['controls_ms'])

path_st = '/home/vant/code/tfm1/data/structural'
path_func = '/home/vant/code/tfm1/data/functional'

csv_files_st = [file for file in sorted(os.listdir(path_st))]
csv_files_func = [file for file in sorted(os.listdir(path_func))]

st_matrices = [pd.read_csv(os.path.join(path_st, file), header=None) for file in csv_files_st]
func_matrices = [pd.read_csv(os.path.join(path_func, file), header=None) for file in csv_files_func]


patient_ids = patient_info['id']

filtered_st_matrices = [st for st, file in zip(st_matrices, csv_files_st) if file.replace('.csv', '') in patient_ids.values]
filtered_func_matrices = [st for st, file in zip(func_matrices, csv_files_func) if file.replace('.csv', '') in patient_ids.values]

print(len(filtered_st_matrices))

controls_list = controls['controls_ms'].tolist()

ms_st = [filtered_st_matrices[i] for i, value in enumerate(controls_list) if value == 1]
hv_st = [filtered_st_matrices[i] for i, value in enumerate(controls_list) if value == 0]

print(len(ms_st))
print(len(hv_st))