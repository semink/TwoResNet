# TwoResNet

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./figs/tworesnet_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="./figs/tworesnet_light.svg">
  <img alt="Shows an illustrated sun in light color mode and a moon with stars in dark color mode." src="https://user-images.githubusercontent.com/25423296/163456779-a8556205-d0a5-45e2-ac17-42d089e3c3f8.png">
</picture>

This is a [PyTorch lightning](https://www.pytorchlightning.ai) implementation of Two-Level resolution Neural Network (TwoResNet) for traffic forecasting.

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

In case error occurs, try to install PyTorch according to your local environment following the description [here](https://pytorch.org/).

## 2. Model training

> You can find tensorboard logs for pretrained models [here](https://tensorboard.dev/experiment/lYEoAwmZQzyrUVT4t4Y8og/).

### 2.1. METR-LA

```bash
python run.py --config=data/config/training.yaml --train --dataset=la
```

### 2.2. PEMS-BAY

```bash
python run.py --config=data/config/training.yaml --train --dataset=bay
```

## 3. Test

### 3.1. METR-LA

```bash
python run.py --config=data/config/test.yaml --test --dataset=la
```

Result

```bash
Horizon 1 (5 min) - MAE: 2.24, RMSE: 3.86, MAPE: 5.32
Horizon 2 (10 min) - MAE: 2.49, RMSE: 4.60, MAPE: 6.19
Horizon 3 (15 min) - MAE: 2.65, RMSE: 5.08, MAPE: 6.78
Horizon 4 (20 min) - MAE: 2.79, RMSE: 5.47, MAPE: 7.29
Horizon 5 (25 min) - MAE: 2.90, RMSE: 5.79, MAPE: 7.73
Horizon 6 (30 min) - MAE: 3.01, RMSE: 6.07, MAPE: 8.14
Horizon 7 (35 min) - MAE: 3.09, RMSE: 6.30, MAPE: 8.47
Horizon 8 (40 min) - MAE: 3.17, RMSE: 6.51, MAPE: 8.78
Horizon 9 (45 min) - MAE: 3.23, RMSE: 6.68, MAPE: 9.05
Horizon 10 (50 min) - MAE: 3.29, RMSE: 6.83, MAPE: 9.28
Horizon 11 (55 min) - MAE: 3.34, RMSE: 6.96, MAPE: 9.50
Horizon 12 (60 min) - MAE: 3.39, RMSE: 7.08, MAPE: 9.71
Aggregation - MAE: 2.97, RMSE: 6.01, MAPE: 8.02
```

### 3.2. PEMS-BAY

```bash
python run.py --config=data/config/test.yaml --test --dataset=bay
```

Result

```bash
Horizon 1 (5 min) - MAE: 0.87, RMSE: 1.56, MAPE: 1.67
Horizon 2 (10 min) - MAE: 1.12, RMSE: 2.21, MAPE: 2.26
Horizon 3 (15 min) - MAE: 1.30, RMSE: 2.73, MAPE: 2.72
Horizon 4 (20 min) - MAE: 1.43, RMSE: 3.14, MAPE: 3.08
Horizon 5 (25 min) - MAE: 1.53, RMSE: 3.45, MAPE: 3.37
Horizon 6 (30 min) - MAE: 1.61, RMSE: 3.69, MAPE: 3.60
Horizon 7 (35 min) - MAE: 1.68, RMSE: 3.88, MAPE: 3.79
Horizon 8 (40 min) - MAE: 1.73, RMSE: 4.03, MAPE: 3.95
Horizon 9 (45 min) - MAE: 1.78, RMSE: 4.15, MAPE: 4.09
Horizon 10 (50 min) - MAE: 1.82, RMSE: 4.25, MAPE: 4.21
Horizon 11 (55 min) - MAE: 1.85, RMSE: 4.33, MAPE: 4.31
Horizon 12 (60 min) - MAE: 1.89, RMSE: 4.41, MAPE: 4.40
Aggregation - MAE: 1.55, RMSE: 3.59, MAPE: 3.45
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
