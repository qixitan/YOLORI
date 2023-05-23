# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/12/2

import torch
from torch import nn
from yolori.models.network_blocks import NonlocalAttention


class MAA(nn.Module):
    def __init__(
            self,
            width=1.0,
            in_features=("dark3", "dark4", "dark5"),
    ):
        super().__init__()
        # self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        in_channels = [256, 512, 1024]
        self.in_features = in_features
        self.unsample = nn.Upsample(scale_factor=2)
        self.maxsample = nn.MaxPool2d(kernel_size=2)
        self.conv = nn.Conv2d(int(sum(in_channels) * width), int(in_channels[1] * width), kernel_size=1, stride=1,
                              padding=0)
        self.conv0 = nn.Conv2d(int((in_channels[1] + in_channels[2]) * width), int(in_channels[2] * width),
                               kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(int((in_channels[1] + in_channels[0]) * width), int(in_channels[0] * width),
                               kernel_size=1, stride=1, padding=0)
        self._nonlocal = NonlocalAttention(int(in_channels[1] * width))

    def forward(self, input):
        #  backbone
        integrates = []
        # out_features = self.backbone(input)
        features = [input[f] for f in self.in_features]
        [x2, x1, x0] = features
        x2_integrate = self.maxsample(x2)
        integrates.append(x2_integrate)
        integrates.append(x1)
        x0_integrate = self.unsample(x0)
        integrates.append(x0_integrate)
        integrate = torch.cat(integrates, dim=1)
        integrate = self.conv(integrate)
        Refin = self._nonlocal(integrate)
        x1 = x1 + Refin
        x2 = self.conv2(torch.cat([self.unsample(Refin), x2], dim=1))
        x0 = self.conv0(torch.cat([self.maxsample(Refin), x0], dim=1))
        outputs = (x2, x1, x0)
        return outputs
