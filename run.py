from model.TwoResNet.dataset import DataModule

from argparse import ArgumentParser
import yaml

from model.TwoResNet.supervisor import Supervisor as TwoResNetSupervisor

from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar, ModelCheckpoint, LearningRateMonitor

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
import numpy as np

import os
from lib import utils

import shortuuid

import torch
from model import const


def data_load(dataset, batch_size=32, seq_len=12, horizon=12, adj_info=dict(sparcity=dict(corr=0.9, prox=0.9)), cluster_info=dict(K=1), time_feat_mode='sinusoidal',
              dow=True, test_on_time=None, num_workers=os.cpu_count(), **kwargs):
    if test_on_time is None:
        test_on_time = (('00:00', '23:55'),)
    dm = DataModule(dataset, batch_size, seq_len,
                    horizon, num_workers, adj_info, cluster_info, time_feat_mode, dow, test_on_time)
    dm.prepare_data()
    return dm


def train_model(config, dataset=None, dparams=None, checkpoint_dir=None, additional_callbacks=None,
                filename_placeholder=''):
    dm = data_load(dataset, **dparams['DATA'], **config['DATA'])

    model = TwoResNetSupervisor(hparams=config, dparams=dparams,
                                input_dim=dm.get_input_dim(),
                                adj_mx=dm.get_adj(),
                                scaler=dm.get_scaler(),
                                cluster_label=dm.get_cluster())

    dparams['LOG']['save_dir'] = os.path.join(
        f"{utils.PROJECT_ROOT}/{dparams['LOG']['save_dir']}", dataset)
    logger = TensorBoardLogger(
        **dparams['LOG'],
        default_hp_metric=False,
        version=f'{filename_placeholder}_{shortuuid.uuid()}_K{config["DATA"]["cluster_info"]["K"]}')

    if checkpoint_dir:
        dparams['TRAINER']["resume_from_checkpoint"] = os.path.join(
            checkpoint_dir, "checkpoint")

    callbacks = [RichModelSummary(**dparams['SUMMARY']),
                 RichProgressBar(),
                 LearningRateMonitor(logging_interval='epoch'),
                 ModelCheckpoint(filename='best',
                                 monitor=f"{dparams['METRIC']['monitor_metric_name']}/{dparams['METRIC']['loss_metric']}",
                                 save_last=True)]

    if additional_callbacks:
        [callbacks.append(callback) for callback in additional_callbacks]
    trainer = Trainer(
        **dparams['TRAINER'], **config['TRAINER'],
        callbacks=callbacks,
        logger=logger)

    trainer.fit(model, dm)


def load_essentials(dataset, dparams):
    checkpoint_dir = dparams['TEST']['checkpoint'][dataset]['dir_path']
    with open(os.path.join(checkpoint_dir, 'hparams.yaml')) as f:
        hparams = yaml.load(f, yaml.FullLoader)['hparams']

    with open(os.path.join(checkpoint_dir, 'cluster.npy'), 'rb') as f:
        cluster_label = np.load(f)

    dm = data_load(dataset, **dparams['DATA'],
                   **hparams['DATA'], test_on_time=dparams['TEST']['on_time'])

    checkpoint = os.path.join(checkpoint_dir+"/checkpoints",
                              dparams['TEST']['checkpoint'][dataset]['file_name'])

    supervisor = TwoResNetSupervisor.load_from_checkpoint(
        checkpoint, dparams=dparams, scaler=dm.get_scaler(),
        input_dim=dm.get_input_dim(),
        adj_mx=dm.get_adj(), cluster_label=cluster_label)

    trainer = Trainer(**dparams['TRAINER'], **hparams['TRAINER'],
                      callbacks=[RichProgressBar()],
                      enable_checkpointing=False,
                      logger=False)
    # trainer.test(model, dm)
    return {'supervisor': supervisor, 'datamodule': dm, 'trainer': trainer}


def test_model(dataset, dparams):
    essentials = load_essentials(dataset, dparams)
    essentials['trainer'].test(
        essentials['supervisor'], essentials['datamodule'])


def predict_model(dataset, dparams):
    essentials = load_essentials(dataset, dparams)
    predictions = essentials['trainer'].predict(
        essentials['supervisor'], essentials['datamodule'])
    return torch.cat(predictions, dim=const.BATCH_DIM)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    # Program specific args
    parser.add_argument("--config", type=str,
                        default="data/config/training.yaml", help="Configuration file path")
    parser.add_argument("--dataset", type=str,
                        default="la", help="name of the dataset. it should be either la or bay.",
                        choices=['la', 'bay'])
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')

    args = parser.parse_args()

    assert (
        not args.train) | (
        not args.test), "Only one of --train and --test flags can be turned on."
    assert (
        args.train) | (
        args.test), "At least one of --train and --test flags must be turned on."

    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)

    if args.train:
        train_model(config['HPARAMS'], args.dataset, config['NONPARAMS'])
    elif args.test:
        test_model(args.dataset, config)
