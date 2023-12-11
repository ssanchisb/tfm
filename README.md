# tfm
On this repository I gather most of the code used in the implementation phase of the TFM project.
The different files under the `code` folder contain the code used for the preprocessing of the data. The results
of the preprocessing are stored in the `data` folder.
Within the tools directory are the functions used for the statistical analysis and the classification task.
The `notebooks` folder contains the notebooks used for the analysis and the classification task.

**Code files:**
- 'equalize_folder_lenghts.py': This file contains the code used to equalize the number of files in the different folders. The
resulting folders are stored in the `data` folder, in the 'functional' and 'structural' folders respectively.
Brain volumes files are also converted to csv and stored in the 'data' folder.
- 'gender_age_correction.py': This script tests for statistical significance of variability in the data sample
due to gender or age.
- harmonize.py and harmonize_func.py: These scripts harmonize the data using ComBat. The resulting data is stored in the
'functional_h' and 'structural_h' folders respectively.
- 'outliers_normalization.py': This script removes outliers and normalizes the data. The resulting data is stored in the
'functional_norm' and 'structural_norm' folders respectively.
- 'filter_edges.py': This script removes values under a 0.1 threshold and eliminates edges that are not present in more than 60% of the controls.
The resulting data is stored in the 'functional_ready' and 'structural_ready' folders respectively, and at that point
the preprocessing stage is finished.





