# Data Science Master's degree - Master's Thesis

---------------
<h2>From brain disconnection to atrophy: Evaluating multi-modal brain network connectivity and regional 
grey matter volumes in Machine Learning-based classification tasks
of Multiple Sclerosis Disease</h2>
---------------
<h3>Paper's abstract</h3>

Departing from sets of brain connectivity data obtained from 147 multiple sclerosis (MS)
patients and 18 healthy volunteer control subjects, we first describe the demographic characteris-
tics of the sample, and then we implement a pre-processing pipeline in order to, subsequently,
determine statistically significant differences between patients and controls in three different
types of measures: 1) Structural and functional connectivity measures in the form of adjacency
matrices of 76x76 nodes (76 brain regions), 2) Graph connectivity measures, both global and
nodal, derived from the same adjacency matrices, and 3) Individual brain volume measures
of the same 76 brain areas. For each of these types of data, the subset of information that
has shown statistically significant differences between groups (patients vs. controls) is fed to a
classification task using a classical Support Vector Machine algorithm (SVM), thus obtaining
different levels of precision in the classification task for each type of data. Results will show that
direct classification using structural FA-weighted connectivity (Fractional Anisotropy) yields
the best precision. However, some graph connectivity measures derived from the same data
attain high levels of precision as well, especially when these are centered on specific brain regions
that seem to be particularly affected by the pathological processes of the disease. Finally, brain
volumes, a measure that portrays the process of brain atrophy, affords the possibility to discern
patients from controls too, albeit with slightly less precision compared to brain connectivity.
We will discuss the potential and eventual shortcomings of this approach to draw an integrated
image of the various processes involved in the pathophysiogical course of multiple sclerosis.

(Full Paper can be found in the `paper` folder of this repository)

------------------------
<h3>Repository Guide</h3>

On this repository we gather the code used in the implementation phase of the project.
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








