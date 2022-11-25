
import os
from argparse import ArgumentParser
import yaml
from pytorch_lightning import Trainer

from ray import tune
from ray.tune import CLIReporter

from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import MedianStoppingRule
from ray.tune.suggest.basic_variant import BasicVariantGenerator


from lib import utils
from lib import tune as tune_lib

from run import train_model


def tune_model(dataset, dparams, hparams):
    config, report_columns = tune_lib.convert_to_tune_config(hparams)
    search_alg = BasicVariantGenerator()

    scheduler = MedianStoppingRule(**dparams['RAY_TUNE']['MedianStoppingRule'])

    reporter = CLIReporter(metric_columns={"loss": dparams['METRIC']['loss_metric'],
                                           "training_iteration": "epoch"},
                           parameter_columns=report_columns,)

    log_dir = os.path.join(
        f"{utils.PROJECT_ROOT}/{dparams['LOG']['save_dir']}", dataset)
    additional_callbacks = [TuneReportCheckpointCallback(metrics={
        "loss": f"{dparams['METRIC']['monitor_metric_name']}/{dparams['METRIC']['loss_metric']}"},
        filename="checkpoint", on="validation_end")]
    analysis = tune.run(
        tune.with_parameters(
            train_model,
            dataset=dataset,
            dparams=dparams,
            additional_callbacks=additional_callbacks),
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
                        default="data/config/tune.yaml", help="Configuration file path")
    parser.add_argument("--dataset", type=str,
                        default="la", help="name of the dataset. it should be either la or bay.",
                        choices=['la', 'bay'])
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)

    tune_model(
        dataset=args.dataset,
        dparams=config['NONPARAMS'],
        hparams=config['HPARAMS']
    )