import torch.nn as nn
import torch
from icosph_nn import utils

class IcosphereConv(nn.Module):
    def __init__(self, out_channels, in_channels):
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=5)

    def forward(self, x):
        assert(len(x.shape) == 3)
        N, C, V = x.shape
        L = utils.get_icosphere_level(V)
        S = 2 * (2**L)
    