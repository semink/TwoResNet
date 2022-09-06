import torch
import torch.nn as nn


class GRUCell(nn.Module):
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

    def forward(self, inputs, hx):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """

        concat_input = torch.cat([inputs, hx], dim=1)
        r = torch.sigmoid(self.linear_r(concat_input))
        z = torch.sigmoid(self.linear_z(concat_input))
        n = self.linear_in(inputs) + r * self.linear_hn(hx)
        n = torch.tanh(n)
        new_state = (1.0 - z) * n + z * hx
        return new_state
