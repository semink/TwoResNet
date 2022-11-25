import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import model.RNN.model as rnn


class LowResNet(nn.Module):
    """Some Information about LowResNet"""

    def __init__(self, in_feat, hid_feat, num_RU, I, out_feat=1, dropout=0.0, **kwargs):
        super(LowResNet, self).__init__()
        self.encoder = rnn.Encoder(recurrent_unit=lambda in_feat, out_feat, dropout: rnn.GRU(in_feat=in_feat, out_feat=out_feat, dropout=dropout, g=lambda i, o: nn.Conv1d(i, o, 1)),
                                   L=num_RU, in_feat=in_feat, out_feat=hid_feat, dropout=dropout)
        self.decoder = rnn.Decoder(recurrent_unit=lambda in_feat, out_feat, dropout: rnn.GRU(in_feat=in_feat, out_feat=out_feat, dropout=dropout, g=lambda i, o: nn.Conv1d(i, o, 1)),
                                   L=num_RU, in_feat=out_feat, hid_feat=hid_feat, out_feat=out_feat, dropout=dropout)
        self.I = I
        self.DI = F.normalize(I, p=1, dim=1)

    def coarse(self, x):
        return self.upscale(self.downscale(x))

    def downscale(self, x):
        mean = torch.einsum('...ns, cn-> ...cs', x, self.DI.type_as(x))
        return mean

    def upscale(self, x):
        spread = torch.einsum('...cs, nc-> ...ns', x, self.I.T.type_as(x))
        return spread

    def forward(self, x, horizon, teacher=None):
        x = self.downscale(x)
        z = self.encoder(x)
        # TODO: This is different to the original implementation. In the original implementation, y0 = zero tensor.
        # y0 = torch.zeros_like(x[..., [0], :, -1])
        y0 = x[..., [0], :, -1]
        y = self.decoder(y0=y0, h0=z, horizon=horizon,
                         teacher=teacher)
        y = self.upscale(y)
        return y
