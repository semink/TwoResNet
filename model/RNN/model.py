import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import const

import numpy as np
from copy import deepcopy as c


class GRU(nn.Module):
    """Some Information about GRU"""

    def __init__(self, in_feat, out_feat, g, dropout=0.0):
        super(GRU, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        do = nn.Dropout(dropout)
        self.g = nn.ModuleDict({'rz': nn.Sequential(g(in_feat+out_feat, 2*out_feat), do),
                                'n': nn.Sequential(g(in_feat+out_feat, out_feat), do)})

    def forward(self, x, h):
        rz = torch.sigmoid(self.g['rz'](torch.concat([x, h], const.FEAT_DIM)))
        r = rz[..., :self.out_feat, :]
        z = rz[..., self.out_feat:, :]
        n = torch.tanh(self.g['n'](torch.concat([x, r*h], const.FEAT_DIM)))
        out = (1.0-z)*n+z*h
        return out


class StackedRU(nn.Module):
    """Some Information about StackedRU"""

    def __init__(self, recurrent_unit: object, L: int, in_feat: int,
                 hid_feat: int, dropout: float = 0.0):
        super(StackedRU, self).__init__()
        self.rus = nn.ModuleList([recurrent_unit(in_feat=in_feat, out_feat=hid_feat, dropout=dropout),
                                  *[recurrent_unit(in_feat=hid_feat, out_feat=hid_feat, dropout=dropout)
                                    for _ in range(L-1)]])
        self.hid_feat = hid_feat

    def initialize_hs(self, reference):
        batch_size, _, K = reference.size()
        L = len(self.rus)
        hs = [reference.new_zeros(batch_size, self.hid_feat, K)
              for _ in range(L)]
        return hs

    def forward(self, x, hs=None):
        input = x
        y = list()

        hs = hs if hs else self.initialize_hs(reference=x)
        for h, ru in zip(hs, self.rus):
            out = ru(input, h)
            input = out
            y.append(out)
        return y


class Encoder(nn.Module):
    """Some Information about Encoder"""

    def __init__(self, recurrent_unit, L, in_feat, out_feat, dropout=0.0):
        super(Encoder, self).__init__()

        self.stacked_ru = StackedRU(recurrent_unit=recurrent_unit, L=L, in_feat=in_feat,
                                    hid_feat=out_feat, dropout=dropout)

    def forward(self, x):
        hs = None
        for x_ in torch.unbind(x, dim=const.TEMPORAL_DIM):
            hs = self.stacked_ru(x_, hs)
        return hs


class Decoder(nn.Module):
    """Some Information about Decoder"""

    def __init__(self, recurrent_unit, L, in_feat, hid_feat, out_feat=1, dropout=0.0):
        super(Decoder, self).__init__()
        self.stacked_ru = StackedRU(recurrent_unit=recurrent_unit, L=L, in_feat=in_feat,
                                    hid_feat=hid_feat, dropout=dropout)
        self.linear = nn.Conv1d(hid_feat, out_feat, 1)

    def forward(self, y0, h0, horizon, offset=None, teacher=None):
        hidden_state = h0
        y = y0
        out = []

        for t in range(horizon):
            if offset is not None:
                y = y + offset[..., t]
            # teacher forcing
            if self.training and teacher:
                y = teacher.help(y, t=t)
            hidden_state = self.stacked_ru(y, hidden_state)
            y = self.linear(hidden_state[-1])
            out.append(y)
        return torch.stack(out, dim=const.TEMPORAL_DIM)


class Teacher:
    def __init__(self,  half_life_epoch=13.33, slope_at_half=-0.1425525, **kwargs):
        self.half_life_epoch = half_life_epoch
        self.slope_at_half = slope_at_half

    def update(self, hint, stage: int):
        self.hint = hint
        self.stage = stage

    def generosity(self, stage):
        c = np.exp(-4*self.half_life_epoch*self.slope_at_half)
        x = c/self.half_life_epoch * np.log(c)
        return c / (c + np.exp((stage*x)/c))

    def help(self, x: torch.Tensor, t: int):
        hint_idx = torch.rand_like(x) < self.generosity(self.stage)
        x = (~hint_idx) * x + hint_idx * self.hint[..., t]
        return x
