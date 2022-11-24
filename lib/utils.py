import numpy as np
import os
import pickle
import scipy.sparse as sp

from pathlib import Path
import wget
import pickle
import os
import pandas as pd
import numpy as np
import torch

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering


def get_project_root() -> Path:
    return Path(__file__).parent.parent


PROJECT_ROOT = get_project_root()


def double_transition_matrix(adj_mx):
    supports = []
    supports.append(torch.tensor(
        calculate_random_walk_matrix(adj_mx)))
    supports.append(torch.tensor(
        calculate_random_walk_matrix(adj_mx.T)))
    return supports


def calculate_random_walk_matrix(adj_mx):
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx)
    return random_walk_mx


def _exist_dataset_on_disk(dataset):
    file = f'{PROJECT_ROOT}/data/METR-LA.csv' if dataset == 'la' else f'{PROJECT_ROOT}/data/PEMS-BAY.csv'
    return os.path.isfile(file)


def split_data(df, rule=[0.7, 0.1, 0.2]):
    assert np.isclose(np.sum(
        rule), 1.0), f"sum of split rule should be 1 (currently sum={np.sum(rule):.2f})"

    num_samples = df.shape[0]
    num_test = round(num_samples * rule[-1])
    num_train = round(num_samples * rule[0])
    num_val = num_samples - num_test - num_train

    train_df = df.iloc[:num_train].copy()
    valid_df = df.iloc[num_train: num_train + num_val].copy()
    test_df = df.iloc[-num_test:].copy()

    return train_df, valid_df, test_df


def get_traffic_data(dataset, null_value=0.0):
    if dataset == 'la':
        fn, adj_name = 'METR-LA.csv', 'adj_mx_METR-LA.pkl'
    elif dataset == 'bay':
        fn, adj_name = 'PEMS-BAY.csv', 'adj_mx_PEMS-BAY.pkl'
    else:
        raise ValueError("dataset name should be either 'bay' or 'la")
    data_url = f'https://zenodo.org/record/5724362/files/{fn}'
    sup_url = f'https://zenodo.org/record/5724362/files/{adj_name}'
    if not _exist_dataset_on_disk(dataset):
        wget.download(data_url, out=f'{PROJECT_ROOT}/data')
        wget.download(sup_url, out=f'{PROJECT_ROOT}/data')
    df = pd.read_csv(f'{PROJECT_ROOT}/data/{fn}', index_col=0)
    df.index = pd.DatetimeIndex(df.index)
    # dt = pd.Timedelta(df.index.to_series().diff().mode().values[0])
    # df = df.asfreq(freq=dt, fill_value=null_value)
    df = df.replace(0.0, null_value)
    with open(f'{PROJECT_ROOT}/data/{adj_name}', 'rb') as f:
        _, _, adj = pickle.load(f, encoding='latin1')
    return df, adj


def convert_timestamp_to_feature(timestamp):
    hour, minute = timestamp.hour, timestamp.minute
    feature = (hour * 60 + minute) / (24 * 60)
    return pd.DataFrame(feature, index=timestamp)


def apply_mask(data, mask):
    data[mask, ...] = 0
    return data


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def torch_mask_null(input, null_value=0.0, eps=1e-3):
    mask = ~torch.isnan(input) if np.isnan(
        null_value) else ~((input <= null_value + eps) & (-input >= null_value - eps))
    return mask


def masked_metric(agg_fn, error_fn, pred, target, null_value=0.0, agg_dim=0):
    mask = (target != null_value).float()
    target_ = target.clone()
    target_[mask == 0.0] = 1.0  # for mape
    mask /= torch.mean(mask, dim=agg_dim, keepdim=True)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    score = error_fn(pred, target_)
    score = score*mask
    # score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
    return agg_fn(score)


def masked_MAE(pred, target, null_value=0.0, agg_dim=(0, 1, 2)):
    mae = masked_metric(agg_fn=lambda e: torch.mean(e, dim=agg_dim),
                        error_fn=lambda p, t: torch.absolute(p - t),
                        pred=pred, target=target, null_value=null_value, agg_dim=agg_dim)
    return mae


def masked_MSE(pred, target, null_value=0.0, agg_dim=(0, 1, 2)):
    mse = masked_metric(agg_fn=lambda e: torch.mean(e, dim=agg_dim),
                        error_fn=lambda p, t: (p - t) ** 2,
                        pred=pred, target=target, null_value=null_value, agg_dim=agg_dim)
    return mse


def masked_RMSE(pred, target, null_value=0.0, agg_dim=(0, 1, 2)):
    rmse = masked_metric(agg_fn=lambda e: torch.sqrt(torch.mean(e, dim=agg_dim)),
                         error_fn=lambda p, t: (p - t)**2,
                         pred=pred, target=target, null_value=null_value, agg_dim=agg_dim)
    return rmse


def masked_MAPE(pred, target, null_value=0.0, agg_dim=(0, 1, 2)):
    mape = masked_metric(agg_fn=lambda e: torch.mean(torch.absolute(e) * 100, dim=agg_dim),
                         error_fn=lambda p, t: ((p - t) / (t)),
                         pred=pred, target=target, null_value=null_value, agg_dim=agg_dim)
    return mape


# spectral clustering

def get_signal_distance(df):
    distance_signal = pdist(df.dropna().T, 'correlation')
    distance_signal = pd.DataFrame(squareform(
        distance_signal), index=df.columns, columns=df.columns)
    return distance_signal


def gaussian_kernel(distance, sparcity=0.9, patience=1):
    nan_portion = get_sparcity(distance, np.nan)
    threshold_cut = np.nanquantile(
        distance, (1-sparcity)/(1-nan_portion) if nan_portion < sparcity else 1)
    variance = np.nanvar(distance)
    nan_idx = np.isnan(distance)
    cut_idx = distance > threshold_cut
    S_dist = np.exp(-distance / (patience * np.sqrt(variance)))
    S_dist[cut_idx] = 0
    S_dist[nan_idx] = 0
    return S_dist


def get_mix_similarity(df_raw, distance_km, sparcity=dict(prox=0.9, corr=0.9), patience=1, alpha=0.5,
                       method='addition', **kwargs):
    sensors = df_raw.columns.astype(int)
    distance_km = distance_km.pivot(
        index='from', columns='to', values='distance').loc[sensors][sensors]
    physical_similarity = distance_km
    signal_similarity = get_signal_distance(df_raw)
    if method == 'addition':
        mix_similarity = alpha * gaussian_kernel(physical_similarity.values,
                                                 sparcity=sparcity['prox'], patience=patience) + (1 - alpha) * gaussian_kernel(signal_similarity.values,
                                                                                                                               sparcity=sparcity['corr'], patience=patience)
    elif method == 'multiplication':
        mix_similarity = gaussian_kernel(physical_similarity.values,
                                         sparcity=sparcity['prox'], patience=patience) * gaussian_kernel(signal_similarity.values,
                                                                                                         sparcity=sparcity['corr'], patience=patience)
    return mix_similarity, sensors


def get_sparcity(mat, nan_val=0.0):
    if np.isnan(nan_val):
        zeros = np.isnan(mat).sum()
    else:
        zeros = (mat == nan_val).sum()
    return zeros/mat.size


def clustering(mix_similarity, sensors, K=5,  **kwargs):
    # 2. Spectral clustering
    clustering = SpectralClustering(n_clusters=K,
                                    assign_labels='discretize',
                                    random_state=0, affinity='precomputed').fit(mix_similarity)

    new_df = pd.DataFrame(data=clustering.labels_,
                          index=sensors)
    return new_df
