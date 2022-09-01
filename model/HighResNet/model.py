import torch
import torch.nn as nn

from lib import utils
from copy import deepcopy as c


def clones(module, N):
    return nn.ModuleList([c(module) for _ in range(N)])


class RNNEncoder(nn.Module):

    def __init__(self, rnn_cell, n_hid):
        super(RNNEncoder, self).__init__()
        self.rnn_cell = c(rnn_cell)
        self.n_hid = n_hid

    def forward(self, x, topologies):
        batch, seq_len, _, n_sensors = x.size()
        hidden_state = [x.new_zeros(batch, self.n_hid, n_sensors)
                        for _ in range(len(self.rnn_cell.stacked_ru))]
        for seq in range(seq_len):
            xt = x[..., seq, :, :]
            hidden_state = self.rnn_cell(xt, hidden_state, topologies)
        return hidden_state


class RNNDecoder(nn.Module):

    def __init__(self, rnn_cell, linear, horizon: int):
        super(RNNDecoder, self).__init__()
        assert horizon > 0, 'horizon should be bigger than 0'
        self.rnn_cell = c(rnn_cell)
        self.linear = c(linear)
        self.horizon = horizon

    def forward(self, y0, h0, dybar, topologies, dybar_answer_sheet=None, answer_sheet=None, show_answer_p=0):
        # teacher forcing should work only during training
        if not self.training:
            answer_sheet = None
        ht = h0
        ytm1 = y0
        yts = []
        for h in range(self.horizon):
            # if teacher tells the answer, use the answer instead of guessing it... soft teacher forcing
            if (dybar_answer_sheet is not None) and show_answer_p > 0:
                pass_label = torch.rand_like(ytm1) < show_answer_p
                ytm1 = (~pass_label)*(ytm1 + dybar[..., h, :, :])\
                    + pass_label * (answer_sheet[..., h, :, :] +
                                    dybar_answer_sheet[..., h, :, :])
            else:
                ytm1 = ytm1 + dybar[..., h, :, :]
            ht = self.rnn_cell(ytm1, ht, topologies)
            yt = self.linear(ht[-1])
            yts.append(yt)

            ytm1 = yt
        return torch.stack(yts, -3)


class RNNCell(nn.Module):
    def __init__(self, recurrent_unit, nlayers, in_dim, hid_dim):
        super(RNNCell, self).__init__()
        if nlayers > 1:
            self.stacked_ru = nn.ModuleList([*clones(recurrent_unit(in_dim, hid_dim), 1),
                                             *clones(recurrent_unit(hid_dim, hid_dim), nlayers-1)])
        else:
            self.stacked_ru = clones(recurrent_unit(in_dim, hid_dim), 1)

    def forward(self, xt, hidden_state, topologies):
        hidden_states = []
        next_hidden_state = xt
        for hs, ru in zip(hidden_state, self.stacked_ru):
            next_hidden_state = ru(next_hidden_state, hs, topologies)
            hidden_states.append(next_hidden_state)
        return hidden_states


class MultiAdjGNN(nn.Module):
    def __init__(self, gnns, in_dim, out_dim,
                 order, bias_start=0.0, dropout=0):
        super(MultiAdjGNN, self).__init__()
        num_supports = len(gnns)
        self.gnns = gnns
        self.linear = nn.Conv1d(
            in_dim * (order * num_supports + 1), out_dim, 1)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, bias_start)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjs):
        x = torch.cat([x,  # add GNN with zero order
                       *[gnn(x, A) for gnn, A in zip(self.gnns, adjs)]], dim=-2)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class GCN(nn.Module):
    """Some Information about GCN"""

    def __init__(self, order, operation='...fn, mn -> ...fm'):
        super(GCN, self).__init__()
        self.order = order
        self.conv_operation = operation

    def gconv(self, x, A):
        return torch.einsum(self.conv_operation, x, A.type_as(x))

    def forward(self, x, A):
        out = []
        x0 = x
        x1 = self.gconv(x0, A)
        out.append(x1)
        for _ in range(2, self.order + 1):
            x2 = 2 * self.gconv(x1, A) - x0
            out.append(x2)
            x1, x0 = x2, x1
        x = torch.cat(out, dim=-2)
        return x


class GCGRU(nn.Module):
    def __init__(self, in_dim, hid_dim, gcn_nlayers, dropout):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()

        # support other nonlinearities up here?

        gcns = nn.ModuleList([GCN(gcn_nlayers, operation='...fn, mn -> ...fm')
                              for _ in range(2)])
        self.gcn_ru = MultiAdjGNN(gcns,
                                  in_dim=in_dim + hid_dim,
                                  out_dim=hid_dim * 2,
                                  order=gcn_nlayers,
                                  bias_start=1.0,
                                  dropout=dropout)

        self.gcn_C = MultiAdjGNN(gcns,
                                 in_dim=in_dim + hid_dim,
                                 out_dim=hid_dim,
                                 order=gcn_nlayers,
                                 dropout=dropout)
        self.hid_dim = hid_dim

    def forward(self, inputs, hx, topologies):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """

        concat_input = torch.cat([inputs, hx], dim=-2)
        value = torch.sigmoid(self.gcn_ru(concat_input, topologies))
        r = value[:, :self.hid_dim, ...]
        u = value[:, self.hid_dim:, ...]
        C = torch.tanh(self.gcn_C(
            torch.cat([inputs, r * hx], dim=-2), topologies))
        new_state = u * hx + (1.0 - u) * C
        return new_state


class HighResNet(nn.Module):

    def __init__(self, encoder, decoder, topology):
        super(HighResNet, self).__init__()
        self.topologies = utils.double_transition_matrix(topology)
        self.encoder = c(encoder)
        self.decoder = c(decoder)

    def forward(self, x, dybar, answer_sheet=None, dybar_answer_sheet=None, show_answer_p=lambda: False):
        z = self.encoder(x, self.topologies)
        go_sym = torch.zeros_like(x[..., -1, [0], :])
        y = self.decoder(y0=go_sym, h0=z, topologies=self.topologies,
                         dybar=dybar, dybar_answer_sheet=dybar_answer_sheet, answer_sheet=answer_sheet, show_answer_p=show_answer_p)
        return y
