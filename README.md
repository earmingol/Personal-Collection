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
- [Search Algorithms](./Search-Algorithms/):
    - [Breadth First Search](./Search-Algorithms/BreadthFirstSearch.py)
    - [Depth First Search](./Search-Algorithms/DepthFirstSearch.py)
    - Optimization problems:
        - [Genetic Algorithm for finding optimal hyper-parameters of an ANN](./Search-Algorithms/GeneticAlgorithm_ANN_architecture.py)
        - [Integer Linear Programming for finding solution of an optimization of metabolic fluxes](./Search-Algorithms/Metabolic_ILP.py)
        - [Backtracking for optimizing the example of "Wheels Companies Problem"](./Search-Algorithms/Wheels_Companies_problem.py)
        
- [Plots](./Plots/):
    - [Jupyter Notebook with examples of plots made with Plotly](./Plots/Plotly-Plots.ipynb)
    
- [Statistical and Machine Learning Analyses](Stats&MachineLearning/):
    - [Principal Component Analysis](Stats&MachineLearning/PCA/)
    - [Enrichment Analysis](Stats&MachineLearning/Enrichment/)
    - More, coming soon