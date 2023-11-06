import pandas as pd
import os.path
import glob

path = '/home/vant/code/tfm1/data/structural'
csv_files = [file for file in sorted(os.listdir(path))]

st_matrices = [pd.read_csv(os.path.join(path, file), header=None) for file in csv_files]

patient_info = pd.read_csv('/home/vant/code/tfm1/data/clinic.csv', usecols=['id', 'age', 'sex'])

patient_ids = patient_info['id']

filtered_st_matrices = [st for st, file in zip(st_matrices, csv_files) if file.replace('.csv', '') in patient_ids.values]

#print(patient_ids)
print(len(patient_info))
print(len(filtered_st_matrices))

