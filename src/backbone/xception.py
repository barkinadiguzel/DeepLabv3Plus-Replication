import torch
import torch.nn as nn
from blocks.depthwise_sep import DepthwiseSeparableConv

class AlignedXception(nn.Module):
    def __init__(self, in_channels=3, output_stride=16):
        super().__init__()
        if output_stride == 16:
            self.entry_dil = 1
            self.middle_dil = 1
            self.exit_dil = 2
        elif output_stride == 8:
            self.entry_dil = 1
            self.middle_dil = 2
            self.exit_dil = 4

        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = DepthwiseSeparableConv(64, 128, stride=2, dilation=self.entry_dil)
        self.block2 = DepthwiseSeparableConv(128, 256, stride=2, dilation=self.entry_dil)
        self.block3 = DepthwiseSeparableConv(256, 728, stride=2, dilation=self.entry_dil)

        self.middle_blocks = nn.Sequential(*[
            DepthwiseSeparableConv(728, 728, stride=1, dilation=self.middle_dil) for _ in range(8)
        ])

        self.block_exit1 = DepthwiseSeparableConv(728, 1024, stride=1, dilation=self.exit_dil)
        self.block_exit2 = DepthwiseSeparableConv(1024, 1024, stride=1, dilation=self.exit_dil)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.middle_blocks(x)
        x = self.block_exit1(x)
        x = self.block_exit2(x)
        return x
