import torch
import torch.nn as nn


import model.HighResNet.model as HighResNet
import model.LowResNet.model as LowResNet


class TwoResNet(nn.Module):

    def __init__(self, cluster_handler, topology, in_dim, pred_horizon,   # general settings
                 LowResNet_kwargs,                          # LowResNet settings
                 HighResNet_kwargs,                         # HighResNet settings
                 ):

        super(TwoResNet, self).__init__()
        out_dim = 1
        self.cluster_handler = cluster_handler
        self.high_res_block = HighResNet.HighResNet(encoder=HighResNet.RNNEncoder(rnn_cell=HighResNet.RNNCell(recurrent_unit=lambda id, hd: HighResNet.GCGRU(id, hd, HighResNet_kwargs['max_diffusion_step'], HighResNet_kwargs['dropout']),
                                                                                                              nlayers=HighResNet_kwargs[
                                                                                                                  'num_rnn_layers'],
                                                                                                              in_dim=in_dim, hid_dim=HighResNet_kwargs['rnn_units']),
                                                                                  n_hid=HighResNet_kwargs['rnn_units']),
                                                    decoder=HighResNet.RNNDecoder(rnn_cell=HighResNet.RNNCell(recurrent_unit=lambda id, hd: HighResNet.GCGRU(id, hd, HighResNet_kwargs['max_diffusion_step'], HighResNet_kwargs['dropout']),
                                                                                                              nlayers=HighResNet_kwargs[
                                                                                                                  'num_rnn_layers'],
                                                                                                              in_dim=out_dim, hid_dim=HighResNet_kwargs['rnn_units']),
                                                                                  linear=nn.Conv1d(HighResNet_kwargs['rnn_units'], out_dim, 1), horizon=pred_horizon),
                                                    topology=topology)
        self.low_res_block = LowResNet.LowResNet(cluster_handler=self.cluster_handler,
                                                 encoder=LowResNet.RNNEncoder(rnn_cell=LowResNet.RNNCell(recurrent_unit=lambda id, hd: LowResNet.GRU(id, hd, LowResNet_kwargs['dropout']),
                                                                                                         nlayers=LowResNet_kwargs['num_rnn_layers'], in_dim=in_dim, hid_dim=LowResNet_kwargs['rnn_units']),
                                                                              n_hid=LowResNet_kwargs['rnn_units']),
                                                 decoder=LowResNet.RNNDecoder(rnn_cell=LowResNet.RNNCell(recurrent_unit=lambda id, hd: LowResNet.GRU(id, hd, LowResNet_kwargs['dropout']),
                                                                                                         nlayers=LowResNet_kwargs['num_rnn_layers'], in_dim=out_dim, hid_dim=LowResNet_kwargs['rnn_units']),
                                                                              linear=nn.Conv1d(
                                                     LowResNet_kwargs['rnn_units'], out_dim, 1),
                                                     horizon=pred_horizon))

    def forward(self, x, answer_sheet=None, show_answer_p=dict(low=0, high=0)):
        if not self.training:
            show_answer_p = dict(low=0, high=0)
        y_low = self.low_res_block(
            x, answer_sheet, show_answer_p['low'])
        last_input_low = self.cluster_handler.downscale(
            x[..., -1, [0], :].unsqueeze(1))
        dybar = self.cluster_handler.upscale(
            torch.cat([last_input_low, y_low], -3).diff(dim=-3))
        dybar_answer_sheet = self.cluster_handler.upscale(
            torch.cat([last_input_low, self.cluster_handler.downscale(answer_sheet)], -3).diff(dim=-3)) if answer_sheet is not None else None
        y = self.high_res_block(x, dybar, answer_sheet,
                                dybar_answer_sheet, show_answer_p['high'])
        return y, y_low
