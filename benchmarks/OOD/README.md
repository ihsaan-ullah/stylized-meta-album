
This code is adapted from https://github.com/facebookresearch/BalancingGroups to support SMA datasets and Deephyper Bayesian optimization library.

### Installing dependencies
Easiest way to have a working environment for this repo is to create a conda environement with the following commands

```bash
conda env create -f environment.yaml
conda activate balancinggroups
```	

If conda is not available, please install the dependencies listed in the requirements.txt file.

### Download, extract and Generate metadata for datasets

This script downloads, extracts and formats the datasets metadata so that it works with the rest of the code out of the box.

```bash
python setup_datasets.py --download --data_path data
```

### Launch jobs

To reproduce the experiments in the paper : 

```bash
python3 train.py --config config.yaml
```

If you want to run the jobs localy, omit the --partition argument.



