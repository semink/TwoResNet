import torch
import numpy as np

from model.HighResNet.model import HighResNet
from lib.utils import masked_MAE, masked_RMSE, masked_MAPE
import pytorch_lightning as pl

from model.RNN.model import Teacher
from model import const


class Supervisor(pl.LightningModule):
    def __init__(self, adj_mx, scaler, hparams, dparams, input_dim):
        super().__init__()
        self._HighResNet_kwargs = hparams.get('HIGH_RES_NET')
        self._LowResNet_kwargs = hparams.get('LOW_RES_NET')
        self._tf_kwargs = hparams.get('TEACHER_FORCING')
        self._optim_kwargs = hparams.get('OPTIMIZER')
        self._metric_kwargs = dparams.get('METRIC')
        # data set
        self.standard_scaler = scaler

        # Teacher forcing
        self.teacher = Teacher(**self._tf_kwargs)
        self.pred_horizon = hparams['DATA']['horizon']

        # setup model
        self.model = HighResNet(
            in_feat=input_dim, A=adj_mx, **self._HighResNet_kwargs)

        # optimizer setting
        self.example_input_array = torch.rand(
            hparams['DATA']['batch_size'], input_dim, adj_mx.size(0), hparams['DATA']['seq_len'])
        self.save_hyperparameters('hparams')

    def on_train_start(self):
        self.monitor_metric_name = self._metric_kwargs['monitor_metric_name']
        self.training_metric_name = self._metric_kwargs['training_metric_name']
        self.logger.log_hyperparams(
            self.hparams.hparams, {
                f'{self.monitor_metric_name}/mae': 0})

    def forward(self, x):
        y = self.model(x, self.pred_horizon)
        return y

    def predict_rolling_horizon(self, x, horizon):
        y = self.model(x, horizon)
        return self.standard_scaler.inverse_transform(y)

    def validation_step(self, batch, idx):
        x, y = batch
        pred = self.forward(x)
        return {'true': y, 'pred': pred}

    def validation_epoch_end(self, outputs):
        true = torch.cat([output['true']
                          for output in outputs], dim=const.BATCH_DIM)
        pred = torch.cat([output['pred']
                          for output in outputs], dim=const.BATCH_DIM)
        loss = self._compute_all_loss(true, pred)
        for metric in loss:
            self.log_dict(
                {f'{self._metric_kwargs["monitor_metric_name"]}/{metric}': loss[metric], "step": float(self.current_epoch)}, prog_bar=True)
        self.log_dict({f'{self._metric_kwargs["monitor_metric_name"]}/combine':
                       torch.tensor([loss[metric] for metric in loss]).sum(), "step": float(self.current_epoch)}, prog_bar=True)

    def training_step(self, batch, idx):
        epoch_float = float(self.current_epoch) + float(idx) / \
            float(self.trainer.num_training_batches)

        x, y = batch
        horizon = y.size(const.TEMPORAL_DIM)

        # update teachers
        self.teacher.update(hint=y, stage=epoch_float)
        # feedforward
        if self._tf_kwargs['teacher'] == True:
            output = self.model(x, horizon, teacher=self.teacher)
        else:
            output = self.model(x, horizon)

        # highresnet loss
        loss = self._compute_loss(y, output)

        # log
        self.log(f'{self.training_metric_name}/mae',
                 loss, prog_bar=True)
        return loss

    def predict_step(self, batch, idx):
        x, y = batch
        pred = self.forward(x)
        return {'pred': self.standard_scaler.inverse_transform(pred), 'true': self.standard_scaler.inverse_transform(y)}

    def test_step(self, batch, idx):
        x, y = batch
        pred = self.forward(x)
        return {'true': y, 'pred': pred}

    def test_epoch_end(self, outputs):
        true = torch.cat([output['true']
                          for output in outputs], dim=const.BATCH_DIM)
        pred = torch.cat([output['pred']
                          for output in outputs], dim=const.BATCH_DIM)
        loss = self._compute_all_loss(true, pred, agg_dim=(
            const.BATCH_DIM, const.FEAT_DIM, const.SPATIAL_DIM))

        # error for each horizon
        self.pred_results = pred.cpu()
        self.true_values = true.cpu()
        self.test_loss = {metric: loss[metric].cpu() for metric in loss}
        self.agg_losses = self._compute_all_loss(
            self.true_values, self.pred_results)

    def print_test_result(self):
        for h in range(len(self.test_loss["mae"])):
            print(f"Horizon {h+1} ({5*(h+1)} min) - ", end="")
            print(f"MAE: {self.test_loss['mae'][h]:.2f}", end=", ")
            print(f"RMSE: {self.test_loss['rmse'][h]:.2f}", end=", ")
            print(f"MAPE: {self.test_loss['mape'][h]:.2f}")
            if self.logger:
                for m in self.test_loss:
                    self.logger.experiment.add_scalar(
                        f"Test/{m}", self.test_loss[m][h], h)

        # aggregated error
        print("Aggregation - ", end="")
        print(f"MAE: {self.agg_losses['mae']:.2f}", end=", ")
        print(f"RMSE: {self.agg_losses['rmse']:.2f}", end=", ")
        print(f"MAPE: {self.agg_losses['mape']:.2f}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),  **self._optim_kwargs['adam'])

        # dynamic milestones
        milestones = self._get_milestones()
        for i, m in enumerate(milestones):
            self.logger.experiment.add_scalar(
                f'{self._metric_kwargs["training_metric_name"]}/milestones', m, i)

        for epoch in range(milestones[-1]):
            generosity = self.teacher.generosity(epoch)
            self.logger.experiment.add_scalar(
                f'{self._metric_kwargs["training_metric_name"]}/teacher_forcing_probability', generosity, epoch)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.MultiStepLR(
            optimizer, gamma=self._optim_kwargs['multisteplr']['gamma'], milestones=milestones),
            'name': 'learning_rate'}

        return [optimizer], [lr_scheduler]

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), gradient_clip_val)

    def _get_milestones(self):
        init = self._epoch_reach_to_p(self._tf_kwargs['milestone_start_p'])
        diff = np.max([int(init/self._optim_kwargs['multisteplr']
                           ['reduce_step_factor']), 1])*np.arange(0, 4)
        return (init + diff).tolist()

    def _epoch_reach_to_p(self, p):
        for epoch in range(1000):
            if self.teacher.generosity(epoch) < p:
                break
        return epoch-1

    def _compute_loss(self, y_true, y_predicted, agg_dim=(const.BATCH_DIM, const.FEAT_DIM, const.SPATIAL_DIM, const.TEMPORAL_DIM), scale=True):
        if scale:
            y_predicted = self.standard_scaler.inverse_transform(y_predicted)
            y_true = self.standard_scaler.inverse_transform(y_true)
        return masked_MAE(y_predicted, y_true, agg_dim=agg_dim)

    def _compute_all_loss(self, y_true, y_predicted, agg_dim=(const.BATCH_DIM, const.FEAT_DIM, const.SPATIAL_DIM, const.TEMPORAL_DIM), scale=True):
        if scale:
            y_predicted = self.standard_scaler.inverse_transform(y_predicted)
            y_true = self.standard_scaler.inverse_transform(y_true)
        return {'mae': masked_MAE(y_predicted, y_true, agg_dim=agg_dim),
                'rmse': masked_RMSE(y_predicted, y_true, agg_dim=agg_dim),
                'mape': masked_MAPE(y_predicted, y_true, agg_dim=agg_dim)}

    def calculate_loss_and_print_result(self, true, pred, scale=False):
        loss = self._compute_all_loss(true, pred, agg_dim=(
            const.BATCH_DIM, const.FEAT_DIM, const.SPATIAL_DIM), scale=scale)
        loss_dict = {metric: loss[metric].cpu() for metric in loss}
        for h in range(len(loss_dict["mae"])):
            print(f"Horizon {h+1} ({5*(h+1)} min) - ", end="")
            print(f"MAE: {loss_dict['mae'][h]:.2f}", end=", ")
            print(f"RMSE: {loss_dict['rmse'][h]:.2f}", end=", ")
            print(f"MAPE: {loss_dict['mape'][h]:.2f}")

        agg_losses = self._compute_all_loss(
            true, pred, scale=scale)
        # aggregated error
        print("Aggregation - ", end="")
        print(f"MAE: {agg_losses['mae']:.2f}", end=", ")
        print(f"RMSE: {agg_losses['rmse']:.2f}", end=", ")
        print(f"MAPE: {agg_losses['mape']:.2f}")
