import torch
import torch.nn as nn

import model.RNN.model as rnn
import model.GCN.model as gcn

from lib import utils


class HighResNet(nn.Module):
    """Some Information about HighResNet"""

    def __init__(self, in_feat, hid_feat, num_RU, A, gcn_order=2, out_feat=1, dropout=0.0, **kwargs):
        super(HighResNet, self).__init__()
        self.As = utils.double_transition_matrix(A)

        def encoder_recurrent_unit(in_feat, out_feat, dropout):
            return rnn.GRU(in_feat=in_feat, out_feat=out_feat, dropout=dropout,
                           g=lambda i, o: gcn.GCN(i, o, order=gcn_order, As=self.As))

        def decoder_recurrent_unit(in_feat, out_feat, dropout):
            return rnn.GRU(in_feat=in_feat, out_feat=out_feat, dropout=dropout,
                           g=lambda i, o: gcn.GCN(i, o, order=gcn_order, As=self.As))

        self.encoder = rnn.Encoder(recurrent_unit=encoder_recurrent_unit,
                                   L=num_RU, in_feat=in_feat, out_feat=hid_feat, dropout=dropout)
        self.decoder = rnn.Decoder(recurrent_unit=decoder_recurrent_unit,
                                   L=num_RU, in_feat=out_feat, hid_feat=hid_feat, out_feat=out_feat, dropout=dropout)

    def forward(self, x, horizon, offset=None, teacher=None):
        z = self.encoder(x)

        # y0 = torch.zeros_like(x[..., [0], :, -1])
        y0 = x[..., [0], :, -1]
        y = self.decoder(y0=y0, h0=z, horizon=horizon, offset=offset,
                         teacher=teacher)
        return y
