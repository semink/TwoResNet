import torch
import numpy as np
import os

from model.TwoResNet.model import TwoResNet
from lib.utils import masked_MAE, masked_RMSE, masked_MAPE
import pytorch_lightning as pl
import torch.nn.functional as F


class Cluster:
    def __init__(self, label):
        self.label = label
        self.I = self.get_indicator_matrix()
        self.DI = F.normalize(self.I, p=1, dim=0)
        self.num_clusters = int(np.max(self.label)+1)

    def get_indicator_matrix(self):
        I = (self.label[:, None] == np.arange(
            0, np.max(self.label)+1)).astype(int)
        return torch.tensor(I).float()

    def downscale(self, x):
        x = torch.einsum('...fn, nk-> ...fk', x, self.DI.type_as(x))
        return x

    def upscale(self, x):
        x = torch.einsum('...fk, nk-> ...fn', x, self.I.type_as(x))
        return x


class Supervisor(pl.LightningModule):
    def __init__(self, adj_mx, scaler, cluster_label, hparams, dparams, input_dim):
        super().__init__()
        self._HighResNet_kwargs = hparams.get('HighResNet')
        self._LowResNet_kwargs = hparams.get('LowResNet')
        self._optim_kwargs = hparams.get('OPTIMIZER')
        self._tf_kwargs = hparams.get('TEACHER_FORCING')
        self._metric_kwargs = dparams.get('METRIC')
        self.cluster_handler = Cluster(cluster_label)
        # data set
        self.standard_scaler = scaler
        self.tf_HighResNet = bool(
            self._tf_kwargs['mode'].get('HighResNet', False))
        self.tf_LowResNet = bool(
            self._tf_kwargs['mode'].get('LowResNet', False))

        # setup model
        self.model = TwoResNet(
            self.cluster_handler,
            adj_mx, input_dim, hparams['DATA']['horizon'],  self._LowResNet_kwargs, self._HighResNet_kwargs
        )
        self.monitor_metric_name = self._metric_kwargs['monitor_metric_name']
        self.training_metric_name = self._metric_kwargs['training_metric_name']

        # optimizer setting
        seq_len = np.max(
            [self._HighResNet_kwargs['seq_len'], self._LowResNet_kwargs['seq_len']])
        self.example_input_array = torch.rand(
            hparams['DATA']['batch_size'], seq_len, input_dim, adj_mx.size(0))

        # teacher forcing
        self.sampling_p = lambda epoch: self._compute_sampling_threshold(
            epoch, **self._tf_kwargs['probability'])

        self.save_hyperparameters('hparams')

    def on_train_start(self):
        self.logger.log_hyperparams(
            self.hparams.hparams, {
                f'{self.monitor_metric_name}/mae': 0})
        for cluster in range(self.cluster_handler.num_clusters):
            self.logger.experiment.add_scalar(f'{self.training_metric_name}/sensors_per_each_clusters', float(
                (self.cluster_handler.label == cluster).sum()), float(cluster))
        with open(os.path.join(self.logger.log_dir, 'cluster.npy'), 'wb') as f:
            np.save(f, self.cluster_handler.label)

    def forward(self, x):
        y, _ = self.model(x)
        return y

    def validation_step(self, batch, idx):
        x, y = batch
        pred = self.forward(x)
        return {'true': y, 'pred': pred}

    def validation_epoch_end(self, outputs):
        true = torch.cat([output['true'] for output in outputs], dim=0)
        pred = torch.cat([output['pred'] for output in outputs], dim=0)
        losses = self._compute_all_loss(true, pred, dim=(0, 1, 2, 3))
        loss = {'mae': losses[0],
                'rmse': losses[1],
                'mape': losses[2]}
        for metric in loss:
            self.log_dict(
                {f'{self.monitor_metric_name}/{metric}': loss[metric], "step": float(self.current_epoch)}, prog_bar=True)

    def training_step(self, batch, idx):
        epoch_float = float(self.current_epoch) + float(idx) / \
            float(self.trainer.num_training_batches)
        sampling_p = self.sampling_p(epoch_float)
        self.log_dict(
            {f'{self.training_metric_name}/teacher_forcing_probability': sampling_p, "step": epoch_float})
        x, y = batch

        show_answer_p = {'low': sampling_p if self.tf_LowResNet else 0,
                         'high': sampling_p if self.tf_HighResNet else 0}
        output, output_general = self.model(x, y, show_answer_p=show_answer_p)
        loss_detail = self._compute_loss(y, output)
        y_general = self.cluster_handler.downscale(y)
        loss_agg = self._compute_loss(
            y_general, output_general)
        self.log(f'{self.training_metric_name}/mae',
                 loss_detail, prog_bar=True)
        self.log(f'{self.training_metric_name}/mae_agg',
                 loss_agg, prog_bar=True)
        loss = loss_detail + \
            self._optim_kwargs['LowResNet_loss_weight'] * loss_agg
        return loss

    def test_step(self, batch, idx):
        x, y = batch
        pred = self.forward(x)
        return {'true': y, 'pred': pred}

    def test_epoch_end(self, outputs):
        true = torch.cat([output['true'] for output in outputs], dim=0)
        pred = torch.cat([output['pred'] for output in outputs], dim=0)
        losses = self._compute_all_loss(true, pred)
        loss = {'mae': losses[0],
                'rmse': losses[1],
                'mape': losses[2]}

        # error for each horizon
        for h in range(len(loss["mae"])):
            print(f"Horizon {h+1} ({5*(h+1)} min) - ", end="")
            print(f"MAE: {loss['mae'][h]:.2f}", end=", ")
            print(f"RMSE: {loss['rmse'][h]:.2f}", end=", ")
            print(f"MAPE: {loss['mape'][h]:.2f}")
            if self.logger:
                for m in loss:
                    self.logger.experiment.add_scalar(
                        f"Test/{m}", loss[m][h], h)

        # aggregated error
        agg_losses = self._compute_all_loss(true, pred, dim=(0, 1, 2, 3))
        print("Aggregation - ", end="")
        print(f"MAE: {agg_losses[0]:.2f}", end=", ")
        print(f"RMSE: {agg_losses[1]:.2f}", end=", ")
        print(f"MAPE: {agg_losses[2]:.2f}")

        self.pred_results = pred.cpu()
        self.true_values = true.cpu()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),  **self._optim_kwargs['adam'])

        # dynamic milepost
        milestones = self._get_milestones()
        for i, m in enumerate(milestones):
            self.logger.experiment.add_scalar(
                f'{self.training_metric_name}/milestones', m, i)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.MultiStepLR(
            optimizer, gamma=self._optim_kwargs['multisteplr']['gamma'], milestones=milestones),
            'name': 'learning_rate'}

        return [optimizer], [lr_scheduler]

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
        torch.nn.utils.clip_grad_norm_(
            self.model.high_res_block.parameters(), gradient_clip_val)
        torch.nn.utils.clip_grad_norm_(
            self.model.low_res_block.parameters(), gradient_clip_val)

    def _get_milestones(self):
        init = self._epoch_reach_to_p(self._tf_kwargs['milestone_start_p'])
        diff = np.max([int(init/self._optim_kwargs['multisteplr']
                           ['reduce_step_factor']), 1])*np.arange(0, 4)
        return (init + diff).tolist()

    def _epoch_reach_to_p(self, p):
        for epoch in range(1000):
            if self.sampling_p(epoch) < p:
                break
        return epoch-1

    @ staticmethod
    def _compute_sampling_threshold(epoch, half_life_epoch=13.33, slope_at_half=-0.1425525):
        c = np.exp(-4*half_life_epoch*slope_at_half)
        x = c/half_life_epoch * np.log(c)
        return c / (c + np.exp((epoch*x)/c))

    @ staticmethod
    def teacher_forcing(sampling_p, curricular_learning_flag):
        go_flag = (torch.rand(1).item() < sampling_p) & (
            curricular_learning_flag)
        return go_flag

    def _compute_loss(self, y_true, y_predicted, dim=(0, 1, 2, 3)):
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        y_true = self.standard_scaler.inverse_transform(y_true)
        return masked_MAE(y_predicted, y_true, dim=dim)

    def _compute_all_loss(self, y_true, y_predicted, dim=(0, 2, 3)):
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        y_true = self.standard_scaler.inverse_transform(y_true)
        return (masked_MAE(y_predicted, y_true, dim=dim),
                masked_RMSE(y_predicted, y_true, dim=dim),
                masked_MAPE(y_predicted, y_true, dim=dim))
