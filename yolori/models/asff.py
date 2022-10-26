# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/10/24

import torch
from torch import nn
import torch.nn.functional as F
from .darknet import CSPDarknet
from .network_blocks import BaseConv


class ASFF(nn.Module):
    def __init__(self, depth=1.0, width=1.0, out_features=("dark3", "dark4", "dark5"), in_channels=[256, 512, 1024],
                 depthwise=False, rfb=False, vis=False, act="leakyrelu"):
        super(ASFF, self).__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_channels = [int(x * width) for x in in_channels[-len(out_features):]]
        self.in_features = out_features
        # self.stride_level_0_1 = BaseConv(in_channels=256, out_channels=self.inter_dim, kernel_size=3, stride=2)
        # self.stride_level_0_2 = BaseConv(in_channels=256, out_channels=self.inter_dim, kernel_size=3, stride=2)
        compress_c = 8 if rfb else 16
        self.vis = vis

        # dark5
        self.stride_level_5_4 = BaseConv(in_channels=self.in_channels[1], out_channels=self.in_channels[2],
                                         kernel_size=3, stride=2, act=act)
        self.stride_level_5_3 = BaseConv(in_channels=self.in_channels[0], out_channels=self.in_channels[2],
                                         kernel_size=3, stride=2, act=act)
        self.weight_level_5 = BaseConv(compress_c * 3, 3, 1, 1, act=act)
        self.weight_level_5_5 = BaseConv(self.in_channels[2], compress_c, 1, 1, act=act)
        self.weight_level_5_4 = BaseConv(self.in_channels[2], compress_c, 1, 1, act=act)
        self.weight_level_5_3 = BaseConv(self.in_channels[2], compress_c, 1, 1, act=act)
        self.expend_5 = BaseConv(self.in_channels[2], self.in_channels[2], kernel_size=3, stride=1, act=act)

        # dark4
        self.stride_level_4_5 = BaseConv(in_channels=self.in_channels[2], out_channels=self.in_channels[1],
                                         kernel_size=1, stride=1, act=act)
        self.stride_level_4_3 = BaseConv(in_channels=self.in_channels[0], out_channels=self.in_channels[1],
                                         kernel_size=1, stride=2, act=act)
        self.weight_level_4 = BaseConv(compress_c * 3, 3, 1, 1, act=act)
        self.weight_level_4_5 = BaseConv(self.in_channels[1], compress_c, 1, 1, act=act)
        self.weight_level_4_4 = BaseConv(self.in_channels[1], compress_c, 1, 1, act=act)
        self.weight_level_4_3 = BaseConv(self.in_channels[1], compress_c, 1, 1, act=act)
        self.expend_4 = BaseConv(self.in_channels[1], self.in_channels[1], kernel_size=3, stride=1, act=act)

        # dark3
        self.stride_level_3_5 = BaseConv(in_channels=self.in_channels[2], out_channels=self.in_channels[0],
                                         kernel_size=1, stride=1, act=act)
        self.stride_level_3_4 = BaseConv(in_channels=self.in_channels[1], out_channels=self.in_channels[0],
                                         kernel_size=1, stride=1, act=act)
        self.weight_level_3 = BaseConv(compress_c * 3, 3, 1, 1, act=act)
        self.weight_level_3_5 = BaseConv(self.in_channels[0], compress_c, 1, 1, act=act)
        self.weight_level_3_4 = BaseConv(self.in_channels[0], compress_c, 1, 1, act=act)
        self.weight_level_3_3 = BaseConv(self.in_channels[0], compress_c, 1, 1, act=act)
        self.expend_3 = BaseConv(self.in_channels[0], self.in_channels[0], kernel_size=3, stride=1, act=act)

    def forward(self, inputs):
        out_features = self.backbone(inputs)
        features = [out_features[f] for f in self.in_features]
        [x3, x4, x5] = features

        x5_3 = F.interpolate(self.stride_level_3_5(x5), scale_factor=4, mode="nearest")
        x4_3 = F.interpolate(self.stride_level_3_4(x4), scale_factor=2, mode="nearest")
        x3_3 = x3

        level_weight_3 = F.softmax(self.weight_level_3(
            torch.cat([self.weight_level_3_3(x3_3), self.weight_level_3_4(x4_3), self.weight_level_3_5(x5_3)], dim=1)),
            dim=1)
        asff_out3 = x3_3 * level_weight_3[:, 0:1, :, :] + \
                    x4_3 * level_weight_3[:, 1:2, :, :] + \
                    x5_3 * level_weight_3[:, 2:, :, :]

        x5_4 = F.interpolate(self.stride_level_4_5(x5), scale_factor=2, mode="nearest")
        x4_4 = x4
        x3_4 = self.stride_level_4_3(x3)
        level_weight_4 = F.softmax(self.weight_level_4(
            torch.cat([self.weight_level_4_3(x3_4), self.weight_level_4_4(x4_4), self.weight_level_4_5(x5_4)], dim=1)),
            dim=1)
        asff_out4 = x3_4 * level_weight_4[:, 0:1, :, :] + \
                    x4_4 * level_weight_4[:, 1:2, :, :] + \
                    x5_4 * level_weight_4[:, 2:, :, :]

        x5_5 = x5
        x4_5 = self.stride_level_5_4(x4)
        x3_5 = self.stride_level_5_3(F.max_pool2d(x3, 3, stride=2, padding=1))
        level_weight_5 = F.softmax(self.weight_level_5(
            torch.cat([self.weight_level_5_3(x3_5), self.weight_level_5_4(x4_5), self.weight_level_5_5(x5)], dim=1)),
            dim=1)
        asff_out5 = x3_5 * level_weight_5[:, 0:1, :, :] + \
                    x4_5 * level_weight_5[:, 1:2, :, :] + \
                    x5_5 * level_weight_5[:, 2:, :, :]

        return (asff_out3, asff_out4, asff_out5)
