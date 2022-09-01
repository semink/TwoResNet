from model.TwoResNet.dataset import DataModule

from argparse import ArgumentParser
import yaml

from model.TwoResNet.supervisor import Supervisor as TwoResNetSupervisor

from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
import numpy as np

import os


def data_load(dataset, batch_size, seq_len, horizon, cluster_info, time_feat_mode, day_of_week,
              test_on_time=None, num_workers=os.cpu_count(), **kwargs):
    if test_on_time is None:
        test_on_time = (('00:00', '23:55'),)
    dm = DataModule(dataset, batch_size, seq_len,
                    horizon, num_workers, cluster_info, time_feat_mode, day_of_week, test_on_time)
    dm.prepare_data()
    return dm


def train_model(dataset, dparams, hparams):
    seq_len = np.max(
        [hparams['HighResNet']['seq_len'], hparams['LowResNet']['seq_len']])
    dm = data_load(dataset, **dparams['DATA'],
                   **hparams['DATA'], seq_len=seq_len)
    model = TwoResNetSupervisor(hparams=hparams,
                                dparams=dparams,
                                input_dim=dm.get_input_dim(),
                                adj_mx=dm.get_adj(),
                                scaler=dm.get_scaler(),
                                cluster_label=dm.get_cluster())

    dparams['TENSORBOARD_LOG']['save_dir'] = os.path.join(
        dparams['TENSORBOARD_LOG']['save_dir'], dataset)

    logger = TensorBoardLogger(
        **dparams['TENSORBOARD_LOG'],
        default_hp_metric=False)

    trainer = Trainer(
        **dparams['TRAINER'], **hparams['TRAINER'],
        # max_epochs=model._get_milestones()[-2],
        callbacks=[RichModelSummary(**dparams['SUMMARY']),
                   RichProgressBar(),
                   LearningRateMonitor(logging_interval='epoch'),
                   ModelCheckpoint(filename='best',
                                   monitor=f"{dparams['METRIC']['monitor_metric_name']}/mae", save_last=True),
                   EarlyStopping(
                       monitor=f"{dparams['METRIC']['monitor_metric_name']}/mae", **dparams['EARLY_STOPPING'])
                   ],
        logger=logger)

    trainer.fit(model, dm)
    result = trainer.test(model, dm, ckpt_path='best')


def test_model(dataset, dparams):
    checkpoint_dir = dparams['TEST']['checkpoint'][dataset]['dir_path']
    with open(os.path.join(checkpoint_dir, 'hparams.yaml')) as f:
        hparams = yaml.load(f, yaml.FullLoader)['hparams']

    with open(os.path.join(checkpoint_dir, 'cluster.npy'), 'rb') as f:
        cluster_label = np.load(f)

    seq_len = np.max(
        [hparams['HighResNet']['seq_len'], hparams['LowResNet']['seq_len']])
    dm = data_load(dataset, **dparams['DATA'],
                   **hparams['DATA'], test_on_time=dparams['TEST']['on_time'], seq_len=seq_len)

    checkpoint = os.path.join(checkpoint_dir+"/checkpoints",
                              dparams['TEST']['checkpoint'][dataset]['file_name'])
    model = TwoResNetSupervisor.load_from_checkpoint(
        checkpoint, dparams=dparams, scaler=dm.get_scaler(),
        input_dim=dm.get_input_dim(),
        adj_mx=dm.get_adj(), cluster_label=cluster_label)

    trainer = Trainer(**dparams['TRAINER'], **hparams['TRAINER'],
                      callbacks=[RichProgressBar()],
                      enable_checkpointing=False,
                      logger=False)
    result = trainer.test(model, dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    # Program specific args
    parser.add_argument("--config", type=str,
                        default="data/model/TwoResNet.yaml", help="Configuration file path")
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
        train_model(args.dataset, config['NONPARAMS'], config['HPARAMS'])
    elif args.test:
        test_model(args.dataset, config['NONPARAMS'])
