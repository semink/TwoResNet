import torch
import torch.nn as nn

from model.LowResNet.model import LowResNet
from model.HighResNet.model import HighResNet

from model import const


class TwoResNet(nn.Module):
    """Some Information about TwoResNet"""

    def __init__(self, in_feat, LowResNet_kwargs, HighResNet_kwargs):
        super(TwoResNet, self).__init__()
        self.lowresnet = LowResNet(in_feat=in_feat, **LowResNet_kwargs)
        self.highresnet = HighResNet(in_feat=in_feat, **HighResNet_kwargs)

    def forward(self, x, horizon, teachers={'highresnet': None, 'lowresnet': None}):
        ybar = self.lowresnet(x, horizon=horizon,
                              teacher=teachers['lowresnet'])
        ybar_diff = torch.diff(torch.cat([self.lowresnet.coarse(x[..., [0], :, :][..., [-1]]),
                                          ybar], dim=const.TEMPORAL_DIM), dim=const.TEMPORAL_DIM)
        y = self.highresnet(x, horizon=horizon, offset=ybar_diff,
                            teacher=teachers['highresnet'])
        out = (y, ybar)
        return out
        