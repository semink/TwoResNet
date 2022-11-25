import torch
import torch.nn as nn

from model import const


class GCN(nn.Module):
    """Some Information about GCN"""

    def __init__(self, in_feat, out_feat, order, As, operation='...cn, nm -> ...cm'):
        super(GCN, self).__init__()
        self.order = order
        self.conv_operation = operation
        self.linear = nn.Conv1d(
            in_feat * (order * 2 + 1), out_feat, 1)
        self.As = As

    def gconv(self, x, A):
        return torch.einsum(self.conv_operation, x, A.type_as(x))

    def chebyshev_conv(self, x, A):
        out = []
        x0 = x
        x1 = self.gconv(x0, A)
        out.append(x1)
        for _ in range(2, self.order + 1):
            x2 = 2 * self.gconv(x1, A) - x0
            out.append(x2)
            x1, x0 = x2, x1
        x = torch.cat(out, dim=const.FEAT_DIM)
        return x

    def concat_gconv(self, x):
        out = [x]
        for A in self.As:
            out.append(self.chebyshev_conv(x, A))
        return torch.cat(out, dim=const.FEAT_DIM)

    def forward(self, x):
        x = self.concat_gconv(x)
        x = self.linear(x)
        return x
