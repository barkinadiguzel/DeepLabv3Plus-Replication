import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, low_level_channels, decoder_channels=256, num_refine_convs=2):
        super().__init__()
        self.conv_low = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn_low = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)

        layers = []
        in_ch = 48 + decoder_channels
        for _ in range(num_refine_convs):
            layers.append(nn.Conv2d(in_ch, decoder_channels, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(decoder_channels))
            layers.append(nn.ReLU(inplace=True))
            in_ch = decoder_channels
        self.refine = nn.Sequential(*layers)

    def forward(self, x, low_level_feat):
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        low = self.relu(self.bn_low(self.conv_low(low_level_feat)))
        x = torch.cat([x, low], dim=1)
        x = self.refine(x)
        return x
