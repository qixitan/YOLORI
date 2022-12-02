# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/12/2
# fusion neck

import torch
from torch import nn
import torch.nn.functional as F
from yolori.models.network_blocks import SCA, BaseConv, CSPLayer, DWConv
from .asff import ASFF_Block2, ASFF_Block3


class AS_SCA(nn.Module):
    """ SCA + ASFF"""

    def __init__(self, width=1.0, in_features=("dark3", "dark4", "dark5"), compression=1):
        super().__init__()
        self.in_features = in_features
        self.asff3 = ASFF_Block2(level=3, multiplier=width)
        self.asff4 = ASFF_Block2(level=4, multiplier=width)
        self.asff5 = ASFF_Block2(level=5, multiplier=width)

        in_channels = [256, 512, 1024]
        self.unsample = nn.Upsample(scale_factor=2)
        self.maxsample = nn.MaxPool2d(kernel_size=2)

        self.expend_level_5 = BaseConv(int(in_channels[2] * width), int(in_channels[2] * width * compression),
                                       kernel_size=1, stride=1)
        self.compress_level_4 = BaseConv(int(in_channels[1] * width), int(in_channels[1] * width * compression),
                                         kernel_size=1, stride=1)
        self.stride_level_3 = BaseConv(int(in_channels[0] * width), int(in_channels[0] * width * compression),
                                       kernel_size=3, stride=2)
        self.expend = BaseConv(int(sum(in_channels) * width * compression), int(in_channels[1] * width * compression),
                               kernel_size=1, stride=1)

        self.SCA = SCA(int(in_channels[1] * width * compression))

    def forward(self, input):
        features = [input[f] for f in self.in_features]
        [x_level_3, x_level_4, x_level_5] = features
        x_level_3_integrate = self.stride_level_3(x_level_3)
        x_level_4_integrate = self.compress_level_4(x_level_4)
        x_level_5_integrate = F.interpolate(x_level_5, scale_factor=2, mode="nearest")
        x_level_5_integrate = self.expend_level_5(x_level_5_integrate)
        integrate = torch.cat([x_level_3_integrate, x_level_4_integrate, x_level_5_integrate], dim=1)
        integrate = self.expend(integrate)

        # SCA
        sca = self.SCA(integrate)
        # ASFF
        out3 = self.asff3([sca, x_level_3])
        out4 = self.asff4([sca, x_level_4])
        out5 = self.asff5([sca, x_level_5])
        out = (out3, out4, out5)
        return out


class PAFPN_ASFF(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"), depthwise=False, act="silu", ):
        super().__init__()
        self.in_features = in_features
        in_channels = [256, 512, 1024]
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        self.asff3 = ASFF_Block3(level=3, multiplier=width)
        self.asff4 = ASFF_Block3(level=4, multiplier=width)
        self.asff5 = ASFF_Block3(level=5, multiplier=width)

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  from backbone features
        features = [input[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        panoutputs = [pan_out2, pan_out1, pan_out0]
        asff_out3 = self.asff3(panoutputs)
        asff_out4 = self.asff4(panoutputs)
        asff_out5 = self.asff5(panoutputs)
        outputs = (asff_out3, asff_out4, asff_out5)
        return outputs
