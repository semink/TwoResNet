import torch
import numpy as np
import os

from model.TwoResNet.model import TwoResNet
from lib.utils import masked_MAE, masked_RMSE, masked_MAPE
import pytorch_lightning as pl

from model.RNN.model import Teacher
from model import const


class Cluster:
    def __init__(self, label):
        self.label = label
        self.num_clusters = int(np.max(self.label)+1)

    def get_indicator_matrix(self):
        I = (self.label[:, None] == np.arange(
            0, np.max(self.label)+1)).astype(int).T
        return torch.tensor(I).float()


class Supervisor(pl.LightningModule):
    def __init__(self, adj_mx, scaler, cluster_label, hparams, dparams, input_dim):
        super().__init__()
        self._HighResNet_kwargs = hparams.get('HIGH_RES_NET')
        self._LowResNet_kwargs = hparams.get('LOW_RES_NET')
        self._tf_kwargs = hparams.get('TEACHER_FORCING')
        self._optim_kwargs = hparams.get('OPTIMIZER')
        self._metric_kwargs = dparams.get('METRIC')
        # data set
        self.standard_scaler = scaler
        self.cluster_handler = Cluster(cluster_label)

        # Teacher forcing
        self.dummy_teacher = Teacher(**self._tf_kwargs)
        self.teachers = dict(highresnet=Teacher(**self._tf_kwargs) if bool(
            self._tf_kwargs.get('high_resolution_model', False)) else None,
            lowresnet=Teacher(**self._tf_kwargs) if bool(
            self._tf_kwargs.get('low_resolution_model', False)) else None)

        self.pred_horizon = hparams['DATA']['horizon']

        # setup model
        self.model = TwoResNet(in_feat=input_dim,
                               LowResNet_kwargs={
                                   **self._LowResNet_kwargs, 'I': self.cluster_handler.get_indicator_matrix()},
                               HighResNet_kwargs={**self._HighResNet_kwargs, 'A': adj_mx})

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
        for cluster in range(self.cluster_handler.num_clusters):
            self.logger.experiment.add_scalar(f'{self.training_metric_name}/sensors_per_each_clusters', float(
                (self.cluster_handler.label == cluster).sum()), float(cluster))
        with open(os.path.join(self.logger.log_dir, 'cluster.npy'), 'wb') as f:
            np.save(f, self.cluster_handler.label)

    def forward(self, x):
        y, _ = self.model(x, self.pred_horizon)
        return y

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

    def update_teachers(self, x_last, answer, stage):
        # update lowresnet teacher
        if self.teachers['lowresnet'] is not None:
            self.teachers['lowresnet'].update(
                hint=self.model.lowresnet.downscale(answer), stage=stage)

        # update highresnet teacher
        if self.teachers['highresnet'] is not None:
            answer_agg = self.model.lowresnet.downscale(
                torch.cat([x_last, answer], dim=const.TEMPORAL_DIM))
            answer_diff_agg = torch.diff(answer_agg, dim=const.TEMPORAL_DIM)
            answer_diff_coarse = self.model.lowresnet.upscale(answer_diff_agg)
            self.teachers['highresnet'].update(
                hint=answer+answer_diff_coarse, stage=stage)

    def training_step(self, batch, idx):
        epoch_float = float(self.current_epoch) + float(idx) / \
            float(self.trainer.num_training_batches)

        x, y = batch
        horizon = y.size(const.TEMPORAL_DIM)

        # update teachers
        self.update_teachers(
            x_last=x[..., [0], :, :][..., [-1]], answer=y, stage=epoch_float)
        # feedforward
        output, output_coarse = self.model(x, horizon, self.teachers)

        # highresnet loss
        loss_detail = self._compute_loss(y, output)

        # lowresnet loss
        y_agg = self.model.lowresnet.downscale(y)
        output_agg = self.model.lowresnet.downscale(output_coarse)
        loss_agg = self._compute_loss(y_agg, output_agg)

        # log
        self.log(f'{self.training_metric_name}/mae',
                 loss_detail, prog_bar=True)
        self.log(f'{self.training_metric_name}/mae_agg',
                 loss_agg, prog_bar=True)

        # total loss
        loss = loss_detail + \
            self._optim_kwargs['low_resol_loss_weight'] * loss_agg
        return loss

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
            generosity = self.dummy_teacher.generosity(epoch)
            self.logger.experiment.add_scalar(
                f'{self._metric_kwargs["training_metric_name"]}/teacher_forcing_probability', generosity, epoch)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.MultiStepLR(
            optimizer, gamma=self._optim_kwargs['multisteplr']['gamma'], milestones=milestones),
            'name': 'learning_rate'}

        return [optimizer], [lr_scheduler]

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
        torch.nn.utils.clip_grad_norm_(
            self.model.highresnet.parameters(), gradient_clip_val)
        torch.nn.utils.clip_grad_norm_(
            self.model.lowresnet.parameters(), gradient_clip_val)

    def _get_milestones(self):
        init = self._epoch_reach_to_p(self._tf_kwargs['milestone_start_p'])
        diff = np.max([int(init/self._optim_kwargs['multisteplr']
                           ['reduce_step_factor']), 1])*np.arange(0, 4)
        return (init + diff).tolist()

    def _epoch_reach_to_p(self, p):
        for epoch in range(1000):
            if self.dummy_teacher.generosity(epoch) < p:
                break
        return epoch-1

    def _compute_loss(self, y_true, y_predicted, agg_dim=(const.BATCH_DIM, const.FEAT_DIM, const.SPATIAL_DIM, const.TEMPORAL_DIM)):
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        y_true = self.standard_scaler.inverse_transform(y_true)
        return masked_MAE(y_predicted, y_true, agg_dim=agg_dim)

    def _compute_all_loss(self, y_true, y_predicted, agg_dim=(const.BATCH_DIM, const.FEAT_DIM, const.SPATIAL_DIM, const.TEMPORAL_DIM)):
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        y_true = self.standard_scaler.inverse_transform(y_true)
        return {'mae': masked_MAE(y_predicted, y_true, agg_dim=agg_dim),
                'rmse': masked_RMSE(y_predicted, y_true, agg_dim=agg_dim),
                'mape': masked_MAPE(y_predicted, y_true, agg_dim=agg_dim)}
