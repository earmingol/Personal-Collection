# Personal-Collection
Codes to implement and test different algorithms

## Installation of Requirements
To install a conda environment, run:

```bash
conda env create -n EnvName --file Conda-Environments/EnvName.yml
```

To export a new conda environment, run:

```bash
conda activate EnvName
conda env export --no-builds | grep -v "prefix" > Conda-Environments/EnvName.yml
conda deactivate
```

## Table of Contents
- Search Algorithms:
    - Breadth First Search
    - Depth First Search
    - Optimization problems:
        - Genetic Algorithm for finding optimal hyperparameters of an ANN
        - Integer Linear Programming for finding solution of an optimization of metabolic fluxes
        - Backtracking for optimizing the example of "Wheels Companies Problem"
        
- Plots:
    - Jupyter Notebook with examples of plots made with Plotly
    
- Statistical Analyses:
    - Coming soon