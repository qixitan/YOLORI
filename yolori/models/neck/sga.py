# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/12/11

import torch
from torch import nn
import torch.nn.functional as F
from yolori.models.network_blocks import NonlocalAttention, BaseConv
from .asff import ASFF_Block2


class CGNonlocalAttention(nn.Module):
    """
    Channel guidance Nonlocal Attention
    """
    def __init__(self, width=1.0, in_features=("dark3", "dark4", "dark5"),):
        super().__init__()
        # self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        in_channels = [256, 512, 1024]
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
        x2_c = self.conv3(F.max_pool2d(x2, kernel_size=2))
        x1_c = self.conv4(x1)
        x0_c = self.conv5(F.interpolate(x0, scale_factor=2, mode="nearest"))
        integrates = (x2_c + x1_c + x0_c) / 3
        integrates = self.attention(integrates)
        out3 = self.out3(F.interpolate(integrates, scale_factor=2, mode="nearest"))+x2
        out4 = self.out4(integrates)+x1
        out5 = self.out5(F.max_pool2d(integrates, kernel_size=2))+x0
        out = (out3, out4, out5)
        return out


class SGNonlocalAttention(nn.Module):
    " Space guidance Nonlocal Attention "

    def __init__(self, width=1.0, in_features=("dark3", "dark4", "dark5"), compression=1):
        super().__init__()
        self.in_features = in_features

        in_channels = [256, 512, 1024]

        self.conv3 = nn.Conv2d(int(
            in_channels[0] * width), int(in_channels[1] * width), kernel_size=1, stride=1, padding=0
        )
        self.conv4 = nn.Conv2d(
            int(in_channels[1] * width), int(in_channels[1] * width), kernel_size=1, stride=1, padding=0
        )
        self.conv5 = nn.Conv2d(
            int(in_channels[2] * width), int(in_channels[1] * width), kernel_size=1, stride=1, padding=0
        )

        self.attention = NonlocalAttention(int(in_channels[1] * width * compression))

        self.asff3 = ASFF_Block2(level=3, multiplier=width)
        self.asff4 = ASFF_Block2(level=4, multiplier=width)
        self.asff5 = ASFF_Block2(level=5, multiplier=width)

    def forward(self, input):
        features = [input[f] for f in self.in_features]
        [x2, x1, x0] = features
        x2_c = self.conv3(F.max_pool2d(x2, kernel_size=2))
        x1_c = self.conv4(x1)
        x0_c = self.conv5(F.interpolate(x0, scale_factor=2, mode="nearest"))
        integrates = (x2_c + x1_c + x0_c) / 3
        integrates = self.attention(integrates)

        # NolocalAttention
        attention_f = self.attention(integrates)
        # ASFF
        out3 = self.asff3([attention_f, x2])
        out4 = self.asff4([attention_f, x1])
        out5 = self.asff5([attention_f, x0])
        out = (out3, out4, out5)
        return out