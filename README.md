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

> You can find tensorboard logs for pretrained models [here](https://tensorboard.dev/experiment/q2igppHyRV2HqUgPA74x2w/).

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
Horizon 1 (5 min) - MAE: 2.24, RMSE: 3.85, MAPE: 5.40
Horizon 2 (10 min) - MAE: 2.49, RMSE: 4.57, MAPE: 6.19
Horizon 3 (15 min) - MAE: 2.65, RMSE: 5.06, MAPE: 6.78
Horizon 4 (20 min) - MAE: 2.79, RMSE: 5.46, MAPE: 7.27
Horizon 5 (25 min) - MAE: 2.91, RMSE: 5.79, MAPE: 7.71
Horizon 6 (30 min) - MAE: 3.02, RMSE: 6.08, MAPE: 8.09
Horizon 7 (35 min) - MAE: 3.11, RMSE: 6.33, MAPE: 8.42
Horizon 8 (40 min) - MAE: 3.19, RMSE: 6.55, MAPE: 8.71
Horizon 9 (45 min) - MAE: 3.27, RMSE: 6.74, MAPE: 8.98
Horizon 10 (50 min) - MAE: 3.34, RMSE: 6.91, MAPE: 9.23
Horizon 11 (55 min) - MAE: 3.40, RMSE: 7.07, MAPE: 9.44
Horizon 12 (60 min) - MAE: 3.47, RMSE: 7.22, MAPE: 9.66
Aggregation - MAE: 2.99, RMSE: 6.06, MAPE: 7.99
```

### 3.2. PEMS-BAY

```bash
python run.py --config=data/model/tworesnet.yaml --test --dataset=bay
```

Result

```bash
Horizon 1 (5 min) - MAE: 0.87, RMSE: 1.55, MAPE: 1.68
Horizon 2 (10 min) - MAE: 1.13, RMSE: 2.20, MAPE: 2.26
Horizon 3 (15 min) - MAE: 1.30, RMSE: 2.71, MAPE: 2.70
Horizon 4 (20 min) - MAE: 1.43, RMSE: 3.11, MAPE: 3.04
Horizon 5 (25 min) - MAE: 1.53, RMSE: 3.42, MAPE: 3.32
Horizon 6 (30 min) - MAE: 1.62, RMSE: 3.67, MAPE: 3.56
Horizon 7 (35 min) - MAE: 1.69, RMSE: 3.87, MAPE: 3.77
Horizon 8 (40 min) - MAE: 1.74, RMSE: 4.03, MAPE: 3.95
Horizon 9 (45 min) - MAE: 1.79, RMSE: 4.16, MAPE: 4.10
Horizon 10 (50 min) - MAE: 1.84, RMSE: 4.27, MAPE: 4.24
Horizon 11 (55 min) - MAE: 1.88, RMSE: 4.37, MAPE: 4.35
Horizon 12 (60 min) - MAE: 1.92, RMSE: 4.46, MAPE: 4.47
Aggregation - MAE: 1.56, RMSE: 3.60, MAPE: 3.45
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
