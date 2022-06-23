# TwoResNet

This is [PyTorch lightning](https://www.pytorchlightning.ai) implementation of Two-Level resolution Neural Network (TwoResNet) for traffic forecasting.

## Dependencies

> **_NOTE:_** [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) should be installed in the system.

### Create a conda environment

```bash
conda env create -f environment.yml
```

### Activate the environment

```bash
conda activate tworesnet
```

## Model training

```bash
# METR-LA
python run.py --config_filename=data/model/tworesnet.yaml --train --dataset=la

# PEMS-BAY
python run.py --config_filename=data/model/tworesnet.yaml --train --dataset=bay
```

## Test

```bash
# METR-LA
python run.py --config_filename=data/model/tworesnet.yaml --test --dataset=la

# PEMS-BAY
python run.py --config_filename=data/model/tworesnet.yaml --test --dataset=bay
```

## Citation

If you find this repository, e.g., the code and the datasets, useful in your research, please cite the following paper:

```citation
```
