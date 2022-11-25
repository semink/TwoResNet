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

> You can find tensorboard logs for pretrained models [here](https://tensorboard.dev/experiment/d6JvEPNhQvOmmDVfbP2WZw/).

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
Horizon 1 (5 min) - MAE: 2.24, RMSE: 3.85, MAPE: 5.36
Horizon 2 (10 min) - MAE: 2.49, RMSE: 4.58, MAPE: 6.27
Horizon 3 (15 min) - MAE: 2.66, RMSE: 5.09, MAPE: 6.89
Horizon 4 (20 min) - MAE: 2.80, RMSE: 5.48, MAPE: 7.41
Horizon 5 (25 min) - MAE: 2.91, RMSE: 5.80, MAPE: 7.84
Horizon 6 (30 min) - MAE: 3.02, RMSE: 6.09, MAPE: 8.23
Horizon 7 (35 min) - MAE: 3.10, RMSE: 6.32, MAPE: 8.55
Horizon 8 (40 min) - MAE: 3.18, RMSE: 6.53, MAPE: 8.84
Horizon 9 (45 min) - MAE: 3.25, RMSE: 6.71, MAPE: 9.09
Horizon 10 (50 min) - MAE: 3.31, RMSE: 6.86, MAPE: 9.30
Horizon 11 (55 min) - MAE: 3.36, RMSE: 7.00, MAPE: 9.49
Horizon 12 (60 min) - MAE: 3.41, RMSE: 7.12, MAPE: 9.65
Aggregation - MAE: 2.98, RMSE: 6.03, MAPE: 8.08
```

### 3.2. PEMS-BAY

```bash
python run.py --config=data/config/test.yaml --test --dataset=bay
```

Result

```bash
Horizon 1 (5 min) - MAE: 0.87, RMSE: 1.56, MAPE: 1.67
Horizon 2 (10 min) - MAE: 1.12, RMSE: 2.22, MAPE: 2.26
Horizon 3 (15 min) - MAE: 1.30, RMSE: 2.73, MAPE: 2.70
Horizon 4 (20 min) - MAE: 1.43, RMSE: 3.13, MAPE: 3.06
Horizon 5 (25 min) - MAE: 1.53, RMSE: 3.44, MAPE: 3.35
Horizon 6 (30 min) - MAE: 1.61, RMSE: 3.69, MAPE: 3.59
Horizon 7 (35 min) - MAE: 1.68, RMSE: 3.88, MAPE: 3.79
Horizon 8 (40 min) - MAE: 1.74, RMSE: 4.04, MAPE: 3.97
Horizon 9 (45 min) - MAE: 1.78, RMSE: 4.17, MAPE: 4.12
Horizon 10 (50 min) - MAE: 1.83, RMSE: 4.28, MAPE: 4.25
Horizon 11 (55 min) - MAE: 1.86, RMSE: 4.37, MAPE: 4.36
Horizon 12 (60 min) - MAE: 1.90, RMSE: 4.46, MAPE: 4.47
Aggregation - MAE: 1.55, RMSE: 3.61, MAPE: 3.47
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
