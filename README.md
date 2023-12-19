# tfm
On this repository we gather the code used in the implementation phase of the TFM project.
The different files under the `code` folder contain the code used for the preprocessing of the data. The results
of the preprocessing phase are stored in the `data` folder.
Within the `tools` directory are the functions used for the statistical analysis and the classification task.
Since the classification task is executed on a separate notebook, the functions appear in both the `tools` 
and the `notebooks` folders.
The `notebooks` folder contains the notebooks used for the analysis and the classification task, as well as the 
preliminary demographic analysis of our sample.

We recommend browsing through the files in the following order:

**`Notebooks/demo`:**
- `demo.html`: This notebook contains the demographic analysis of the sample. It is also available in Rmd format.

**`Code` files:**
These are the scripts used to preprocess the data. They apply different transformations to the data, 
and the results are always exported to the `data` folder, so we can track the transformation of the data
at every step.
- `equalize_folder_lenghts.py`: This file contains the code used to equalize the number of files in the different folders. The
resulting folders are stored in the `data` folder, in the `functional` and `structural folders respectively.
Brain volumes files are also converted to csv and stored in the 'data' folder.
- `gender_age_correction.py`: This script tests for statistical significance of variability in the data sample
due to gender or age.
- `harmonize.py` and `harmonize_func.py`: These scripts harmonize the data using ComBat. The resulting data is stored in the
`functional_h` and `structural_h` folders respectively.
- `outliers_normalization.py`: This script removes outliers and normalizes the data. The resulting data is stored in the
`functional_norm` and `structural_norm` folders respectively.
- `filter_edges.py`: This script removes values under a 0.1 threshold and eliminates edges that are not present in more than 60% of the controls.
The resulting data is stored in the `functional_ready` and `structural_ready` folders respectively, and at that point
the preprocessing stage is finished.

**`Notebooks/svm`:** This notebook, available in html and ipynb format,
contains the code used for the classification task. After the preprocessing stage,
this notebook showcases the main body of work for this project, and the results obtained in it
will be the base of our conclusions and discussion.

**`Visualizations`:** Here we find a few images that will help contextualize the results obtained in the classification task.
- `viz.ipynb` and `viz.html`: This notebook contains the code used to generate the visualizations used in the presentation.
- `viz.pdf`: This file contains the visualizations used in the presentation.








