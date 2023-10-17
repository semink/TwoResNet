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
        I = self.cluster_handler.get_indicator_matrix()
        self.model = TwoResNet(in_feat=input_dim,
                               LowResNet_kwargs={
                                   **self._LowResNet_kwargs, 'I': self.cluster_handler.get_indicator_matrix()},
                               HighResNet_kwargs={**self._HighResNet_kwargs,
                                                  'A': adj_mx,
                                                  #   'A': adj_mx * (I.T@I)

                                                  })

        # optimizer setting
        self.example_input_array = torch.rand(
            hparams['DATA']['batch_size'], input_dim, adj_mx.size(0), hparams['DATA']['seq_len'])

        self.best_score = {'mae': np.inf, 'rmse': np.inf,
                           'mape': np.inf, 'combine': np.inf}

        self.save_hyperparameters('hparams')

    def on_train_start(self):
        self.monitor_metric_name = self._metric_kwargs['monitor_metric_name']
        loss_metric = self._metric_kwargs['loss_metric']
        self.training_metric_name = self._metric_kwargs['training_metric_name']
        self.logger.log_hyperparams(
            self.hparams.hparams, {f'{self.monitor_metric_name}/{loss_metric}': 0, f'best/{loss_metric}': 0})
        for cluster in range(self.cluster_handler.num_clusters):
            self.logger.experiment.add_scalar(f'{self.training_metric_name}/sensors_per_each_clusters', float(
                (self.cluster_handler.label == cluster).sum()), float(cluster))
        with open(os.path.join(self.logger.log_dir, 'cluster.npy'), 'wb') as f:
            np.save(f, self.cluster_handler.label)

    def forward(self, x):
        y = self.model(x, self.pred_horizon)
        return y

    def predict_rolling_horizon(self, x, horizon):
        y = self.model(x, horizon)
        return self.standard_scaler.inverse_transform(y)

    def validation_step(self, batch, idx):
        x, y = batch
        pred, _ = self.forward(x)
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

            if loss[metric] <= self.best_score[metric]:
                self.best_score[metric] = loss[metric]
                self.log_dict(
                    {f'best/{metric}': loss[metric], "step": float(self.current_epoch)}, prog_bar=True)

        combine_loss = torch.tensor([loss[metric] for metric in loss]).sum()

        self.log_dict({f'{self._metric_kwargs["monitor_metric_name"]}/combine':
                       combine_loss, "step": float(self.current_epoch)}, prog_bar=True)
        if combine_loss <= self.best_score['combine']:
            self.best_score['combine'] = combine_loss
            self.log_dict({f'best/combine':
                           combine_loss, "step": float(self.current_epoch)}, prog_bar=True)

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

    def predict_step(self, batch, idx):
        x, y = batch
        pred, _ = self.forward(x)
        return {'pred': self.standard_scaler.inverse_transform(pred), 'true': self.standard_scaler.inverse_transform(y)}

    def test_step(self, batch, idx):
        x, y = batch
        pred, pred_agg = self.forward(x)
        return {'true': y, 'pred': pred, 'pred_agg': pred_agg}

    def test_epoch_end(self, outputs):
        true = torch.cat([output['true']
                          for output in outputs], dim=const.BATCH_DIM)
        pred = torch.cat([output['pred']
                          for output in outputs], dim=const.BATCH_DIM)
        pred_agg = torch.cat([output['pred_agg']
                          for output in outputs], dim=const.BATCH_DIM)
        self.calculate_loss_and_print_result(true, pred, scale=True)
        
        self.print_low_high_portion(pred, pred_agg)
    def print_low_high_portion(self, pred, pred_agg):
        pred_scaled = self.standard_scaler.inverse_transform(pred)
        pred_agg_scaled = self.standard_scaler.inverse_transform(pred_agg)
        low_portion = (pred_agg_scaled - pred_scaled).abs().mean(dim=(const.BATCH_DIM, const.SPATIAL_DIM)) / pred_agg_scaled.abs().mean(dim=(const.BATCH_DIM, const.SPATIAL_DIM))
        print(f"High/Low: {low_portion}")
    
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
