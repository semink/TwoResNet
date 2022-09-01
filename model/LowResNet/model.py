import torch
import torch.nn as nn
from copy import deepcopy as c


def clones(module, N):
    return nn.ModuleList([c(module) for _ in range(N)])


class RNNCell(nn.Module):
    def __init__(self, recurrent_unit, nlayers, in_dim, hid_dim):
        super(RNNCell, self).__init__()
        if nlayers > 1:
            self.stacked_ru = nn.ModuleList([*clones(recurrent_unit(in_dim, hid_dim), 1),
                                             *clones(recurrent_unit(hid_dim, hid_dim), nlayers-1)])
        else:
            self.stacked_ru = clones(recurrent_unit(in_dim, hid_dim), 1)

    def forward(self, xt, hidden_state):
        hidden_states = []
        next_hidden_state = xt
        for hs, ru in zip(hidden_state, self.stacked_ru):
            next_hidden_state = ru(next_hidden_state, hs)
            hidden_states.append(next_hidden_state)
        return hidden_states


class RNNEncoder(nn.Module):

    def __init__(self, rnn_cell, n_hid):
        super(RNNEncoder, self).__init__()
        self.rnn_cell = c(rnn_cell)
        self.n_hid = n_hid

    def forward(self, x):
        batch, seq_len, _, n_sensors = x.size()
        hidden_state = [x.new_zeros(batch, self.n_hid, n_sensors)
                        for _ in range(len(self.rnn_cell.stacked_ru))]
        for seq in range(seq_len):
            xt = x[..., seq, :, :]
            hidden_state = self.rnn_cell(xt, hidden_state)
        return hidden_state


class RNNDecoder(nn.Module):

    def __init__(self, rnn_cell, linear, horizon: int):
        super(RNNDecoder, self).__init__()
        assert horizon > 0, 'horizon should be bigger than 0'
        self.rnn_cell = c(rnn_cell)
        self.linear = c(linear)
        self.horizon = horizon

    def forward(self, y0, h0, answer_sheet=None, show_answer_p=0):
        # teacher forcing should work only during training
        if not self.training:
            answer_sheet = None
        ht = h0
        ytm1 = y0
        yts = []
        for h in range(self.horizon):
            if (answer_sheet is not None) and show_answer_p > 0:
                # if teacher tells the answer, use the answer instead of guessing it... soft teacher forcing
                pass_label = torch.rand_like(ytm1) < show_answer_p
                ytm1 = (~pass_label)*ytm1 + pass_label * \
                    answer_sheet[..., h, :, :]
            ht = self.rnn_cell(ytm1, ht)
            yt = self.linear(ht[-1])
            yts.append(yt)

            ytm1 = yt
        return torch.stack(yts, -3)


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


class LowResNet(nn.Module):

    def __init__(self, cluster_handler, encoder, decoder):
        super(LowResNet, self).__init__()
        self.encoder = c(encoder)
        self.decoder = c(decoder)
        self.cluster_handler = cluster_handler

    def forward(self, x, answer_sheet=None, show_answer_p=0):
        x_low = self.cluster_handler.downscale(x)
        if answer_sheet is not None:
            answer_sheet = self.cluster_handler.downscale(answer_sheet)
        z = self.encoder(x_low)
        go_symbol = torch.zeros_like(x_low[..., -1, [0], :])
        y_low = self.decoder(go_symbol, z,
                             answer_sheet=answer_sheet, show_answer_p=show_answer_p)
        return y_low
