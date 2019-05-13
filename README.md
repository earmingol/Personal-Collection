# Personal-Collection
Codes to implement and test different algorithms


To install a conda environment, run:

```bash
conda env create -n EnvName --file Conda-Environments/EnvName.yml
```

To export a new conda environment, run:

```bash
conda env export --no-builds | grep -v "prefix" > Conda-Environments/EnvName.yml
```