
import os
from argparse import ArgumentParser
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar, ModelCheckpoint, LearningRateMonitor
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.basic_variant import BasicVariantGenerator
import ray

from model.TwoResNet.dataset import DataModule
from model.TwoResNet.supervisor import Supervisor as TwoResNetSupervisor

from lib import utils
import shortuuid

import numpy as np

# ray.init(local_mode=True)

common_id = shortuuid.ShortUUID().random(length=3)


def data_load(dataset, batch_size, seq_len, horizon, cluster_info, time_feat_mode, dow,
              test_on_time=None, num_workers=os.cpu_count(), **kwargs):
    if test_on_time is None:
        test_on_time = (('00:00', '23:55'),)
    dm = DataModule(dataset, batch_size, seq_len,
                    horizon, num_workers, cluster_info, time_feat_mode, dow, test_on_time)
    dm.prepare_data()
    return dm


def train_model(config, dataset=None, dparams=None, checkpoint_dir=None):
    seq_len = np.max(
        [config['HIGH_RES_NET']['seq_len'], config['LOW_RES_NET']['seq_len']])
    dm = data_load(dataset, **dparams['DATA'], **config['DATA'],
                   seq_len=seq_len)
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
        version=f'{common_id}_{shortuuid.ShortUUID().random(length=5)}')

    if checkpoint_dir:
        dparams['TRAINER']["resume_from_checkpoint"] = os.path.join(
            checkpoint_dir, "checkpoint")

    trainer = Trainer(
        **dparams['TRAINER'], **config['TRAINER'],
        callbacks=[RichModelSummary(**dparams['SUMMARY']),
                   RichProgressBar(),
                   LearningRateMonitor(logging_interval='epoch'),
                   ModelCheckpoint(filename='best',
                                   monitor=f"{dparams['METRIC']['monitor_metric_name']}/combine", save_last=True),
                   TuneReportCheckpointCallback(metrics={
                                                "loss": f"{dparams['METRIC']['monitor_metric_name']}/combine"},
                                                filename="checkpoint", on="validation_end")
                   ],
        logger=logger)

    trainer.fit(model, dm)


def _tune_config(params, cli_report_columns=[]):
    result = {}
    for key, val in params.items():
        if isinstance(val, dict):       # not an end-point
            val, _ = _tune_config(val, cli_report_columns)
            if not val:                 #
                continue
        if isinstance(val, dict):       #
            result[key] = val
        else:
            if str(val).startswith('tune'):
                result[key] = eval(val)
                cli_report_columns.append(key)
            else:
                result[key] = tune.choice([val])
    return result, cli_report_columns


def tune_asha(dataset, dparams, hparams):
    config, tune_param_names = _tune_config(hparams)
    report_columns = ['/'.join(utils.search(config, tune_param))
                      for tune_param in tune_param_names]
    search_alg = BasicVariantGenerator()

    scheduler = ASHAScheduler(**dparams['RAY_TUNE']['ASHA'])

    reporter = CLIReporter(metric_columns=["loss"],
                           parameter_columns=report_columns,
                           #    parameter_columns=['DATA/cluster_info/K']
                           )

    log_dir = os.path.join(
        f"{utils.PROJECT_ROOT}/{dparams['LOG']['save_dir']}", dataset)
    analysis = tune.run(
        tune.with_parameters(
            train_model,
            dataset=dataset,
            dparams=dparams),
        resources_per_trial={"cpu": 1, "gpu": 1},
        metric="loss",
        mode="min",
        config=config,
        scheduler=scheduler,
        progress_reporter=reporter,
        search_alg=search_alg,
        local_dir=log_dir,
        name="tune_asha",
        **dparams['RAY_TUNE']['RUN'],
    )

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    # Program specific args
    parser.add_argument("--config", type=str,
                        default="data/model/ablation_cluster.yaml", help="Configuration file path")
    parser.add_argument("--dataset", type=str,
                        default="la", help="name of the dataset. it should be either la or bay.",
                        choices=['la', 'bay'])
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)

    # config, cli_report_columnes = _tune_config(config['HPARAMS'])
    tune_asha(
        dataset=args.dataset,
        dparams=config['NONPARAMS'],
        hparams=config['HPARAMS']
    )
