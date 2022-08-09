# TwoResNet

This is [PyTorch lightning](https://www.pytorchlightning.ai) implementation of Two-Level resolution Neural Network (TwoResNet) for traffic forecasting.

## 1. Installing dependencies

### 1.1. Using conda

> **_NOTE:_** [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) should be installed in the system.

#### 1.1.1. Create a conda environment

```bash
conda env create -n tworesnet -f environment.yaml
```

#### 1.1.2. Activate the environment

```bash
conda activate tworesnet
```

### 1.2. Using pip

#### 1.2.1. Create a venv environment

```bash
python3 -m venv env
```

#### 1.2.2. Activate pip environment

```bash
source env/bin/activate
```

#### 1.2.3. Install dependencies

```bash
pip install -r requirements.txt
```

### 1.3. Install PyTorch

In case error occurs, try to install PyTorch according to your local environment following the description here:
<https://pytorch.org/>

## 2. Model training

```bash
# METR-LA
python run.py --config=data/model/tworesnet.yaml --train --dataset=la

# PEMS-BAY
python run.py --config=data/model/tworesnet.yaml --train --dataset=bay
```

## 3. Test

```bash
# METR-LA
python run.py --config=data/model/tworesnet.yaml --test --dataset=la

# PEMS-BAY
python run.py --confige=data/model/tworesnet.yaml --test --dataset=bay
```

## 4. Citation

If you find this repository, e.g., the code and the datasets, useful in your research, please cite the following paper:

```citation
```
