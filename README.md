# TwoResNet

![Alt text](./figs/tworesnet.svg)
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

### 2.1. METR-LA

```bash
python run.py --config=data/model/tworesnet.yaml --train --dataset=la
```

### 2.2. PEMS-BAY

```bash
python run.py --config=data/model/tworesnet.yaml --train --dataset=bay
```

## 3. Test

### 3.1. METR-LA

```bash
python run.py --config=data/model/tworesnet.yaml --test --dataset=la
```

Result

```bash
Horizon 1 (5 min) - MAE: 2.26, RMSE: 3.90, MAPE: 5.50
Horizon 2 (10 min) - MAE: 2.51, RMSE: 4.62, MAPE: 6.26
Horizon 3 (15 min) - MAE: 2.67, RMSE: 5.10, MAPE: 6.84
Horizon 4 (20 min) - MAE: 2.81, RMSE: 5.49, MAPE: 7.30
Horizon 5 (25 min) - MAE: 2.92, RMSE: 5.81, MAPE: 7.72
Horizon 6 (30 min) - MAE: 3.04, RMSE: 6.09, MAPE: 8.11
Horizon 7 (35 min) - MAE: 3.12, RMSE: 6.33, MAPE: 8.41
Horizon 8 (40 min) - MAE: 3.21, RMSE: 6.56, MAPE: 8.71
Horizon 9 (45 min) - MAE: 3.29, RMSE: 6.75, MAPE: 8.97
Horizon 10 (50 min) - MAE: 3.35, RMSE: 6.91, MAPE: 9.18
Horizon 11 (55 min) - MAE: 3.41, RMSE: 7.07, MAPE: 9.40
Horizon 12 (60 min) - MAE: 3.47, RMSE: 7.21, MAPE: 9.61
Aggregation - MAE: 3.01, RMSE: 6.07, MAPE: 8.00
```

### 3. 3.2. PEMS-BAY

### 3.1. METR-LA

```bash
python run.py --config=data/model/tworesnet.yaml --test --dataset=bay
```

Result

```bash
Horizon 1 (5 min) - MAE: 0.87, RMSE: 1.56, MAPE: 1.68
Horizon 2 (10 min) - MAE: 1.13, RMSE: 2.21, MAPE: 2.26
Horizon 3 (15 min) - MAE: 1.31, RMSE: 2.73, MAPE: 2.71
Horizon 4 (20 min) - MAE: 1.44, RMSE: 3.13, MAPE: 3.06
Horizon 5 (25 min) - MAE: 1.54, RMSE: 3.45, MAPE: 3.35
Horizon 6 (30 min) - MAE: 1.63, RMSE: 3.69, MAPE: 3.59
Horizon 7 (35 min) - MAE: 1.70, RMSE: 3.89, MAPE: 3.80
Horizon 8 (40 min) - MAE: 1.76, RMSE: 4.06, MAPE: 3.98
Horizon 9 (45 min) - MAE: 1.81, RMSE: 4.19, MAPE: 4.13
Horizon 10 (50 min) - MAE: 1.85, RMSE: 4.30, MAPE: 4.27
Horizon 11 (55 min) - MAE: 1.89, RMSE: 4.40, MAPE: 4.38
Horizon 12 (60 min) - MAE: 1.93, RMSE: 4.48, MAPE: 4.49
Aggregation - MAE: 1.57, RMSE: 3.62, MAPE: 3.48
```

## 4. Citation

If you find this repository, e.g., the code and the datasets, useful in your research, please cite the following paper:

```citation
@inproceedings{Li2022tworesnet,
      title = {TwoResNet: Two-level resolution neural network for traffic forecasting of freeway networks},
      author = {Li, Danya and Kwak, Semin and Geroliminis, Nikolas},
      year = {2022},
      publisher={25th IEEE International Conference on Intelligent Transportation Systems (ITSC)},
      venue = {Macau, China}, eventdate={2022-10-08/2022-10-12},
}
```
