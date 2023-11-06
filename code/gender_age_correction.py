import pandas as pd
import os.path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm

patient_info = pd.read_csv('/home/vant/code/tfm1/data/clinic.csv', usecols=['id', 'age', 'sex'])

path_st = '/home/vant/code/tfm1/data/structural'
path_func = '/home/vant/code/tfm1/data/functional'

csv_files_st = [file for file in sorted(os.listdir(path_st))]
csv_files_func = [file for file in sorted(os.listdir(path_func))]

st_matrices = [pd.read_csv(os.path.join(path_st, file), header=None) for file in csv_files_st]
func_matrices = [pd.read_csv(os.path.join(path_func, file), header=None) for file in csv_files_func]


patient_ids = patient_info['id']

filtered_st_matrices = [st for st, file in zip(st_matrices, csv_files_st) if file.replace('.csv', '') in patient_ids.values]
filtered_func_matrices = [st for st, file in zip(func_matrices, csv_files_func) if file.replace('.csv', '') in patient_ids.values]

matrix_na_st = [matrix for matrix in filtered_st_matrices if matrix.isnull().values.any()]
matrix_na_func = [matrix for matrix in filtered_func_matrices if matrix.isnull().values.any()]
print(matrix_na_st)
print(matrix_na_func)

# from here on I continue with structural matrices only

combined_data_st = patient_info.copy()
combined_data_st['f_matrix'] = filtered_st_matrices

#print(combined_data_st['f_matrix'].iloc[0])

# Check for missing values:
print(combined_data_st.isnull().sum())
# Check for correct gender encoding:
print(combined_data_st['sex'].unique())

flattened_matrices = [matrix.to_numpy().flatten() for matrix in combined_data_st['f_matrix']]
flattened_data = pd.DataFrame(flattened_matrices)

# Set up data for linear regression:
X = combined_data_st[['age', 'sex']].values
Y = flattened_data.values

model = LinearRegression()
model.fit(X, Y)

predictions = model.predict(X)
r_squared = r2_score(Y, predictions)
mse = mean_squared_error(Y, predictions)

print(f'R-squared: {r_squared}')
print(f'Mean Squared Error: {mse}')

intercept = model.intercept_
coefficients = model.coef_

print(f'Intercept (β0): {intercept}')
print(f'Coefficient for Age (β1): {coefficients[0]}')
print(f'Coefficient for Gender (β2): {coefficients[1]}')



# Assuming you have already trained the linear regression model
# 'model' is the trained model

# Calculate residuals
Y_pred = model.predict(X)
residuals = Y - Y_pred

# Calculate the degrees of freedom
n_samples, n_features = X.shape
dof = n_samples - n_features

# Calculate the standard error of the residuals
stderr = np.sqrt(np.sum(residuals ** 2) / dof)

# Calculate the standard error of the coefficients
coef_stderr = stderr / np.sqrt(np.sum(X ** 2, axis=0))

# Calculate t-values for the coefficients
t_values = model.coef_ / coef_stderr

# Calculate two-tailed p-values based on the t-values
from scipy.stats import t
p_values = 2 * (1 - t.cdf(np.abs(t_values), df=dof))

# 'p_values' now contains the p-values for each coefficient

print("P-values for coefficients:")
print(p_values)
print(type(p_values))
