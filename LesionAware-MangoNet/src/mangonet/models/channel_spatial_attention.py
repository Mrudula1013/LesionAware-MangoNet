"""CBAM-style channel->spatial attention with original-feature residual path."""
import torch
from torch import nn


class ChannelSpatialAttention(nn.Module):
    def __init__(self, channels, channel=True, spatial=True):
        super().__init__()
        self.use_channel, self.use_spatial = channel, spatial
        hidden = max(1, channels // 8)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=False), nn.Linear(hidden, channels)
        ) if channel else None
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False) if spatial else None

    def forward(self, x):
        refined = x
        if self.channel_mlp is not None:
            avg = refined.mean(dim=(2, 3))
            maximum = refined.amax(dim=(2, 3))
            scale = torch.sigmoid(self.channel_mlp(avg) + self.channel_mlp(maximum))
            refined = refined * scale[:, :, None, None]
        if self.spatial_conv is not None:
            avg = refined.mean(dim=1, keepdim=True)
            maximum = refined.amax(dim=1, keepdim=True)
            mask = torch.sigmoid(self.spatial_conv(torch.cat((avg, maximum), dim=1)))
            refined = refined * mask
        return x + refined if (self.use_channel or self.use_spatial) else x
