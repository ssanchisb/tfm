import os.path
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


FA_matrices = [pd.read_csv(file, header=None) for file in
               glob.glob(os.path.join('/home/vant/code/tfm1/data/FA', "*.csv"))]

# Initialize variables to keep track of the matrix with the lowest sum and its sum
lowest_sum_matrix = None
lowest_sum = float('inf')  # Initialize to positive infinity
lowest_sum_index = None

# Iterate through the matrices and calculate their sum of weights
for idx, matrix in enumerate(FA_matrices):
    current_sum = matrix.values.sum()
    if current_sum < lowest_sum:
        lowest_sum = current_sum
        lowest_sum_matrix = matrix
        lowest_sum_index = idx

print("Matrix with the highest total sum of weights:")
print(lowest_sum_matrix)
print("weakest matrix:", lowest_sum_index)

# extract labels of MS vs. HV:
MS_labels = pd.read_csv('/home/vant/code/tfm1/data/demographics.csv')
labels = MS_labels['mstype'].tolist()

print(len(MS_labels))
print(len(labels))
print(labels.count(0))

# use labels to separate into 2 groups:
MS_FA = [FA_matrices[i] for i, value in enumerate(labels) if value == 0]
HV_FA = [FA_matrices[i] for i, value in enumerate(labels) if value == -1]
print(len(MS_FA))
print(len(HV_FA))

# Create average matrices for each of the 2 groups:
avg_fa_ms = pd.concat(MS_FA).groupby(level=0).mean()
avg_fa_hv = pd.concat(HV_FA).groupby(level=0).mean()

# extract a matrix of a single ms patient:
ms1 = pd.read_csv('/home/vant/code/clonemmri/data/FA/0121.csv', header=None)



threshold = 0  # Adjust this value to your desired threshold

avg_fa_ms_masked = avg_fa_ms.where(avg_fa_ms >= threshold, 0)
avg_fa_hv_masked = avg_fa_hv.where(avg_fa_hv >= threshold, 0)
ms1_masked = ms1.where(ms1 >= threshold, 0)


fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Create two subplots side by side

# Plot the first heatmap for MS
sns.heatmap(ms1_masked, cmap='coolwarm', cbar=True, square=False, mask=None, ax=ax[0])
ax[0].xaxis.tick_top()
ax[0].set_title("MS FA Matrix")
ax[0].set_xlabel("Columns")
ax[0].set_ylabel("Rows")

# Plot the second heatmap for HV
sns.heatmap(avg_fa_hv_masked, cmap='coolwarm', cbar=True, square=False, mask=None, ax=ax[1])
ax[1].xaxis.tick_top()
ax[1].set_title("HV FA Matrix")
ax[1].set_xlabel("Columns")
ax[1].set_ylabel("")

plt.show()

