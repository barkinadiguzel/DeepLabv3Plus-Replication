import torch
import torch.nn as nn

class AtrousConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, bias=False):
        super().__init__()
        padding = dilation  
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=padding, dilation=dilation, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
