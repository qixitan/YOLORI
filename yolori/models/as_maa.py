# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/10/24

import torch
from torch import nn
import torch.nn.functional as F
from .darknet import CSPDarknet
from .network_blocks import SCA, BaseConv


class AS_MAA(nn.Module):
    """
     Multi-scale Attention Adaptive model
    """

    def __init__(
            self,
            depth=1.0,
            width=1.0,
            in_features=("dark3", "dark4", "dark5"),
            in_channels=[256, 512, 1024],
            depthwise=False,
            rfb=False,
            act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = [int(x * width) for x in in_channels]
        compress_c = 8 if rfb else 16

        self.stride_level_4_5 = BaseConv(in_channels=self.in_channels[2], out_channels=self.in_channels[1],
                                         kernel_size=1, stride=1, act=act)
        self.stride_level_4_3 = BaseConv(in_channels=self.in_channels[0], out_channels=self.in_channels[1],
                                         kernel_size=1, stride=2, act=act)
        self.weight_level_4 = BaseConv(compress_c * 3, 3, 1, 1, act=act)
        self.weight_level_4_5 = BaseConv(self.in_channels[1], compress_c, 1, 1, act=act)
        self.weight_level_4_4 = BaseConv(self.in_channels[1], compress_c, 1, 1, act=act)
        self.weight_level_4_3 = BaseConv(self.in_channels[1], compress_c, 1, 1, act=act)
        self.expend_4 = BaseConv(self.in_channels[1], self.in_channels[1], kernel_size=3, stride=1, act=act)

        self._nonlocal = SCA(int(in_channels[1] * width))

        # refin和level3
        self.Refin_3 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), kernel_size=3, stride=1,
                                act=act)
        self.weight_Refin_3 = BaseConv(int(in_channels[0] * width), compress_c, kernel_size=1, stride=1, act=act)
        self.weight_3 = BaseConv(int(in_channels[0] * width), compress_c, kernel_size=1, stride=1, act=act)
        self.weight_Refin_level_3 = BaseConv(2 * compress_c, 2, kernel_size=1, stride=1, act=act)
        self.out_expend_3 = BaseConv(self.in_channels[0], self.in_channels[0], kernel_size=3, stride=1, act=act)

        # refin和level4

        self.weight_4 = BaseConv(int(in_channels[1] * width), compress_c, kernel_size=1, stride=1, act=act)
        self.weight_Refin_4 = BaseConv(int(in_channels[1] * width), compress_c, kernel_size=1, stride=1, act=act)
        self.weight_Refin_level_4 = BaseConv(2 * compress_c, 2, kernel_size=1, stride=1, act=act)
        self.out_expend_4 = BaseConv(self.in_channels[1], self.in_channels[1], kernel_size=3, stride=1, act=act)

        # refin和level5
        self.Refin_5 = BaseConv(int(in_channels[1] * width), int(in_channels[2] * width), kernel_size=3, stride=2,
                                act=act)
        self.weight_Refin_5 = BaseConv(int(in_channels[2] * width), compress_c, kernel_size=1, stride=1, act=act)
        self.weight_5 = BaseConv(int(in_channels[2] * width), compress_c, kernel_size=1, stride=1, act=act)
        self.weight_Refin_level_5 = BaseConv(2 * compress_c, 2, kernel_size=1, stride=1, act=act)
        self.out_expend_5 = BaseConv(self.in_channels[2], self.in_channels[2], kernel_size=3, stride=1, act=act)

    def forward(self, input):
        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x3, x4, x5] = features
        #######################
        x5_4 = F.interpolate(self.stride_level_4_5(x5), scale_factor=2, mode="nearest")
        x4_4 = x4
        x3_4 = self.stride_level_4_3(x3)
        level_weight_4 = F.softmax(self.weight_level_4(
            torch.cat([self.weight_level_4_3(x3_4), self.weight_level_4_4(x4_4), self.weight_level_4_5(x5_4)], dim=1)),
            dim=1)
        Refin = x3_4 * level_weight_4[:, 0:1, :, :] + \
                x4_4 * level_weight_4[:, 1:2, :, :] + \
                x5_4 * level_weight_4[:, 2:, :, :]
        Refin = self._nonlocal(self.expend_4(Refin))

        # level_3
        refin_3 = F.interpolate(self.Refin_3(Refin), scale_factor=2, mode="nearest")
        weight_refin_3 = F.softmax(
            self.weight_Refin_level_3(torch.cat([self.weight_3(x3), self.weight_Refin_3(refin_3)], dim=1)), dim=1)
        out3 = self.out_expend_3(x3 * weight_refin_3[:, 0:1, :, :] + refin_3 * weight_refin_3[:, 1:2, :, :])

        # level_4
        weight_refin_4 = F.softmax(
            self.weight_Refin_level_4(torch.cat([self.weight_4(x4), self.weight_Refin_4(Refin)], dim=1)), dim=1)
        out4 = self.out_expend_4(x4 * weight_refin_4[:, 0:1, :, :] + Refin * weight_refin_4[:, 1:2, :, :])

        # level_5
        refin_5 = self.Refin_5(Refin)
        weight_refin_5 = F.softmax(
            self.weight_Refin_level_5(torch.cat([self.weight_Refin_5(refin_5), self.weight_5(x5)], dim=1)), dim=1)
        out5 = self.out_expend_5(refin_5 * weight_refin_5[:, 0:1, :, :] + x5 * weight_refin_5[:, 1:2, :, :])

        outputs = (out3, out4, out5)
        return outputs
