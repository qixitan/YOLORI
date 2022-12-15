# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/12/11

import torch
from torch import nn
import torch.nn.functional as F
from yolori.models.network_blocks import NonlocalAttention


class NoNeck(nn.Module):

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
        self.in_features = in_features
        self.in_channels = in_channels

    def forward(self, input):
        #  from  backbone
        features = [input[f] for f in self.in_features]
        # [x2, x1, x0] = features
        # print(x2.shape, x1.shape, x0.shape)
        return tuple(features)


class NoNeck1(nn.Module):
    """ 使用平衡层"""

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
        self.in_features = in_features
        self.in_channels = in_channels

        self.conv3 = nn.Conv2d(int(
            in_channels[0] * width), int(in_channels[1] * width), kernel_size=1, stride=1, padding=0
        )
        self.conv4 = nn.Conv2d(
            int(in_channels[1] * width), int(in_channels[1] * width), kernel_size=1, stride=1, padding=0
        )
        self.conv5 = nn.Conv2d(
            int(in_channels[2] * width), int(in_channels[1] * width), kernel_size=1, stride=1, padding=0
        )

        self.out3 = nn.Conv2d(
            int(in_channels[1] * width), int(in_channels[0] * width), kernel_size=1, stride=1, padding=0
        )

        self.out4 = nn.Conv2d(
            int(in_channels[1] * width), int(in_channels[1] * width), kernel_size=1, stride=1, padding=0
        )

        self.out5 = nn.Conv2d(
            int(in_channels[1] * width), int(in_channels[2] * width), kernel_size=1, stride=1, padding=0
        )

    def forward(self, input):
        #  backbone
        features = [input[f] for f in self.in_features]

        [x2, x1, x0] = features
        x2 = self.conv3(F.max_pool2d(x2, kernel_size=2))
        x1 = self.conv4(x1)
        x0 = self.conv5(F.interpolate(x0, scale_factor=2, mode="nearest"))
        integrates = (x2 + x1 + x0) / 3

        out3 = self.out3(F.interpolate(integrates, scale_factor=2, mode="nearest"))
        out4 = self.out4(integrates)
        out5 = self.out5(F.max_pool2d(integrates, kernel_size=2))
        out = (out3, out4, out5)
        print(out3.shape, out4.shape, out5.shape)
        return out


class NoNeck2(nn.Module):
    """ 使用平衡层 + nolocal"""

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
        self.in_features = in_features
        self.in_channels = in_channels

        self.conv3 = nn.Conv2d(int(
            in_channels[0] * width), int(in_channels[1] * width), kernel_size=1, stride=1, padding=0
        )
        self.conv4 = nn.Conv2d(
            int(in_channels[1] * width), int(in_channels[1] * width), kernel_size=1, stride=1, padding=0
        )
        self.conv5 = nn.Conv2d(
            int(in_channels[2] * width), int(in_channels[1] * width), kernel_size=1, stride=1, padding=0
        )

        self.attention = NonlocalAttention(int(in_channels[1] * width))

        self.out3 = nn.Conv2d(
            int(in_channels[1] * width), int(in_channels[0] * width), kernel_size=1, stride=1, padding=0
        )

        self.out4 = nn.Conv2d(
            int(in_channels[1] * width), int(in_channels[1] * width), kernel_size=1, stride=1, padding=0
        )

        self.out5 = nn.Conv2d(
            int(in_channels[1] * width), int(in_channels[2] * width), kernel_size=1, stride=1, padding=0
        )

    def forward(self, input):
        #  backbone
        features = [input[f] for f in self.in_features]

        [x2, x1, x0] = features
        x2 = self.conv3(F.max_pool2d(x2, kernel_size=2))
        x1 = self.conv4(x1)
        x0 = self.conv5(F.interpolate(x0, scale_factor=2, mode="nearest"))
        integrates = (x2 + x1 + x0) / 3
        integrates = self.attention(integrates)
        out3 = self.out3(F.interpolate(integrates, scale_factor=2, mode="nearest"))
        out4 = self.out4(integrates)
        out5 = self.out5(F.max_pool2d(integrates, kernel_size=2))
        out = (out3, out4, out5)
        return out
