# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/10/24

import torch
from torch import nn
from .darknet import CSPDarknet
from .network_blocks import NonlocalAttention


class NONeck(nn.Module):

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels

    def forward(self, input):
        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        # [x2, x1, x0] = features
        # outputs = (x2, x1, x0)
        return tuple(features)


class NONeck1(nn.Module):

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels

        self.unsample = nn.Upsample(scale_factor=2)
        self.maxsample = nn.MaxPool2d(kernel_size=2)
        self.conv = nn.Conv2d(int(sum(in_channels)*width), int(in_channels[1]*width), kernel_size=1, stride=1,
                              padding=0)

    def forward(self, input):
        #  backbone
        integrates = []
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]

        [x2, x1, x0] = features
        integrates.append(x1)
        x2_integrate = self.maxsample(x2)
        integrates.append(x2_integrate)
        x0_integrate = self.unsample(x0)
        integrates.append(x0_integrate)
        integrate = torch.cat(integrates, dim=1)
        integrate = self.conv(integrate)
        return tuple([integrate])


class NONeck2(nn.Module):

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels

        self.unsample = nn.Upsample(scale_factor=2)
        self.maxsample = nn.MaxPool2d(kernel_size=2)
        self.conv = nn.Conv2d(int(sum(in_channels)*width), int(in_channels[1]*width), kernel_size=1, stride=1,
                              padding=0)

        self._nonlocal = NonlocalAttention(int(in_channels[1] * width))

    def forward(self, input):
        #  backbone
        integrates = []
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]

        [x2, x1, x0] = features
        integrates.append(x1)
        x2_integrate = self.maxsample(x2)
        integrates.append(x2_integrate)
        x0_integrate = self.unsample(x0)
        integrates.append(x0_integrate)
        integrate = torch.cat(integrates, dim=1)
        integrate = self.conv(integrate)
        Refin = self._nonlocal(integrate)
        return tuple([Refin])
