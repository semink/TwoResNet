import torch
import torch.nn as nn
import torch.nn.functional as F

import copy


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def chebyshev_poly_generator(M, A, skip_0=False):
    assert M >= 0, 'M should be equal or bigger than 0'
    Tnm2 = torch.eye(A.shape[0])
    # 0th order
    if not skip_0:
        yield Tnm2
    # 1st order
    if M >= 1:
        Tnm1 = A
        yield Tnm1
    # M >= 2
    if M >= 2:
        for _ in range(2, M+1):
            Tn = 2*A@Tnm1 - Tnm2
            Tnm2, Tnm1 = Tnm1, Tn
            yield Tn


class GCN(nn.Module):

    def __init__(self, A, M, in_dim, out_dim, dropout, bias_start=0.0):
        super(GCN, self).__init__()
        self.A = torch.Tensor(A)
        self.M = M
        self.linear = nn.Conv1d((2*M+1)*in_dim,  out_dim, 1)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, bias_start)
        self.dropout = nn.Dropout(dropout)
        self.Tns = list(chebyshev_poly_generator(
            self.M, F.normalize(self.A, p=1, dim=1), skip_0=True))
        self.TnTs = list(chebyshev_poly_generator(
            self.M, F.normalize(self.A.T, p=1, dim=1), skip_0=True))

    def gconv(self, x, A):
        device = x.device
        x = torch.einsum('...fn, nm -> ...fm', x, A.to(device))
        return x

    def forward(self, x):
        y = [x]
        for Tn, TnT in zip(self.Tns, self.TnTs):
            y.append(self.gconv(x, Tn))
            y.append(self.gconv(x, TnT))
        y = torch.cat(y, -2)
        y = self.linear(y)
        y = self.dropout(y)
        return y


class GRU(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_r = nn.Sequential(nn.Conv1d(in_dim + hid_dim, hid_dim, 1),
                                      self.dropout)
        self.linear_z = nn.Sequential(
            nn.Conv1d(in_dim + hid_dim, hid_dim, 1), self.dropout)
        self.linear_in = nn.Sequential(
            nn.Conv1d(in_dim, hid_dim, 1), self.dropout)
        self.linear_hn = nn.Sequential(
            nn.Conv1d(hid_dim, hid_dim, 1), self.dropout)

    def forward(self, inputs, htm1):

        concat_input = torch.cat([inputs, htm1], dim=-2)
        r = torch.sigmoid(self.linear_r(concat_input))
        z = torch.sigmoid(self.linear_z(concat_input))
        n = torch.tanh(self.linear_in(inputs) + r * self.linear_hn(htm1))
        ht = (1.0 - z) * n + z * htm1
        return ht


class GCGRU(nn.Module):
    def __init__(self, in_dim, hid_dim, A, gcn_nlayers, dropout):
        super().__init__()
        self.gcn1 = GCN(A, gcn_nlayers, in_dim, hid_dim, dropout)
        self.gcn2 = GCN(A, gcn_nlayers, hid_dim, hid_dim, dropout)
        self.gru = GRU(hid_dim, hid_dim, dropout)

    def forward(self, htlm1, htm1):

        htlm1 = self.gcn1(htlm1)
        htm1 = self.gcn2(htm1)
        ht = self.gru(htlm1, htm1)
        return ht


class RNNCell(nn.Module):

    def __init__(self, recurrent_unit, nlayers, in_dim, hid_dim):
        super(RNNCell, self).__init__()
        if nlayers > 1:
            self.stacked_ru = nn.ModuleList([*clones(recurrent_unit(in_dim, hid_dim), 1),
                                             *clones(recurrent_unit(hid_dim, hid_dim), nlayers-1)])
        else:
            self.stacked_ru = clones(recurrent_unit(in_dim, hid_dim), 1)

    def forward(self, xt, htm1):
        ht = []
        input = xt
        for hs, ru in zip(htm1, self.stacked_ru):
            out = ru(input, hs)
            ht.append(out)
            input = out

        return ht


class RNNEncoder(nn.Module):

    def __init__(self, rnn_cell, n_hid):
        super(RNNEncoder, self).__init__()
        self.rnn_cell = rnn_cell
        self.n_hid = n_hid

    def forward(self, x):
        batch, horizon, _, n_sensors = x.size()
        htm1 = [x.new_zeros(batch, self.n_hid, n_sensors)
                for _ in range(len(self.rnn_cell.stacked_ru))]
        for h in range(horizon):
            xt = x[..., h, :, :]
            ht = self.rnn_cell(xt, htm1)
            htm1 = ht
        return ht


class RNNDecoder(nn.Module):

    def __init__(self, rnn_cell, linear, horizon: int):
        super(RNNDecoder, self).__init__()
        assert horizon > 0, 'horizon should be bigger than 0'
        self.rnn_cell = rnn_cell
        self.linear = linear
        self.horizon = horizon

    def forward(self, y0, h0, dybar=None):
        # teacher forcing should work only during training
        ht = h0
        ytm1 = y0
        yts = []
        for h in range(self.horizon):
            if dybar is not None:
                ytm1 = ytm1 + dybar[..., h, :, :]
            ht = self.rnn_cell(ytm1, ht)
            yt = self.linear(ht[-1])
            yts.append(yt)
            # if teacher tells the answer, use the answer instead of guessing it...
            ytm1 = yt
        return torch.stack(yts, -3)


class LowResNet(nn.Module):

    def __init__(self, I, encoder, decoder):
        super(LowResNet, self).__init__()
        self.I = torch.Tensor(I)
        self.DI = F.normalize(self.I, p=1, dim=0)
        self.encoder = encoder
        self.decoder = decoder

    def downscaling(self, x):
        device = x.device
        x = torch.einsum('...fn, nk-> ...fk', x, self.DI.to(device))
        return x

    def upscaling(self, x):
        device = x.device
        x = torch.einsum('...fk, kn-> ...fn', x, self.I.T.to(device))
        return x

    def forward(self, x):
        x_low = self.downscaling(x)
        z = self.encoder(x_low)
        y_low = self.decoder(x_low[..., -1, [0], :], z)
        ybar = self.upscaling(y_low)
        return ybar


class HighResNet(nn.Module):

    def __init__(self, encoder, decoder):
        super(HighResNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, dybar):
        z = self.encoder(x)
        y = self.decoder(y0=x[..., -1, [0], :], h0=z,
                         dybar=dybar)
        return y


class TwoResNet(nn.Module):

    def __init__(self, I, topology, in_dim, pred_horizon,   # general settings
                 LowResNet_kwargs,                          # LowResNet settings
                 HighResNet_kwargs,                         # HighResNet settings
                 ):

        super(TwoResNet, self).__init__()
        out_dim = 1
        self.high_res_block = HighResNet(encoder=RNNEncoder(rnn_cell=RNNCell(recurrent_unit=lambda in_dim, hid_dim: GCGRU(in_dim, hid_dim, topology, HighResNet_kwargs['max_diffusion_step'], HighResNet_kwargs['dropout']),
                                                                             nlayers=HighResNet_kwargs['num_rnn_layers'], in_dim=in_dim, hid_dim=HighResNet_kwargs['rnn_units']),
                                                            n_hid=HighResNet_kwargs['rnn_units']),
                                         decoder=RNNDecoder(rnn_cell=RNNCell(recurrent_unit=lambda in_dim, hid_dim: GCGRU(in_dim, hid_dim, topology, HighResNet_kwargs['max_diffusion_step'], HighResNet_kwargs['dropout']),
                                                                             nlayers=HighResNet_kwargs['num_rnn_layers'], in_dim=out_dim, hid_dim=HighResNet_kwargs['rnn_units']),
                                                            linear=nn.Conv1d(
                                                                HighResNet_kwargs['rnn_units'], out_dim, 1),
                                                            horizon=pred_horizon))
        self.low_res_block = LowResNet(I,
                                       encoder=RNNEncoder(rnn_cell=RNNCell(recurrent_unit=lambda in_dim, hid_dim: GRU(in_dim, hid_dim, LowResNet_kwargs['dropout']),
                                                                           nlayers=LowResNet_kwargs['num_rnn_layers'], in_dim=in_dim, hid_dim=LowResNet_kwargs['rnn_units']),
                                                          n_hid=LowResNet_kwargs['rnn_units']),
                                       decoder=RNNDecoder(rnn_cell=RNNCell(recurrent_unit=lambda in_dim, hid_dim: GRU(in_dim, hid_dim, LowResNet_kwargs['dropout']),
                                                                           nlayers=LowResNet_kwargs['num_rnn_layers'], in_dim=out_dim, hid_dim=LowResNet_kwargs['rnn_units']),
                                                          linear=nn.Conv1d(
                                                              LowResNet_kwargs['rnn_units'], out_dim, 1),
                                                          horizon=pred_horizon))

    def coarse(self, x):
        return self.low_res_block.upscaling(self.low_res_block.downscaling(x))

    def forward(self, x):
        ybar = self.low_res_block(x)
        last_input = x[..., -1, [0], :].unsqueeze(1)
        dybar = torch.diff(
            torch.cat([self.coarse(last_input), ybar], -3), dim=-3)
        y = self.high_res_block(x, dybar)
        return y, ybar
