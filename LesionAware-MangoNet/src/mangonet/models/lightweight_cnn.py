"""Stride-16 CNN with depthwise separable convolutions and a 3/5/7 multiscale block."""
import torch
from torch import nn


class DWSeparable(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel=3):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel, stride=stride,
                                   padding=kernel // 2, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU(inplace=False)
        self.residual = stride == 1 and in_channels == out_channels

    def forward(self, x):
        y = self.activation(self.bn(self.pointwise(self.depthwise(x))))
        return y + x if self.residual else y


class MultiScale(nn.Module):
    def __init__(self, channels, branch_channels):
        super().__init__()
        self.branches = nn.ModuleList([
            DWSeparable(channels, branch_channels, kernel=k) for k in (3, 5, 7)
        ])
        self.project = nn.Conv2d(branch_channels * 3, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.activation = nn.SiLU(inplace=False)

    def forward(self, x):
        return self.activation(self.bn(self.project(
            torch.cat([branch(x) for branch in self.branches], dim=1)
        )))


class LightweightCNN(nn.Module):
    def __init__(self, channels=(32, 64, 96, 128), branch_channels=64):
        super().__init__()
        a, b, c, d = channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, a, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(a), nn.SiLU(inplace=False)
        )
        self.blocks = nn.ModuleList([
            DWSeparable(a, a),        # 112x112
            DWSeparable(a, b, 2),     # 56x56
            DWSeparable(b, b),
            DWSeparable(b, c, 2),     # 28x28
            DWSeparable(c, c),
            DWSeparable(c, d, 2),     # 14x14
        ])
        self.multiscale = MultiScale(d, branch_channels)

    def forward(self, x):
        x = self.stem(x)
        for layer in self.blocks:
            x = layer(x)
        return self.multiscale(x)
