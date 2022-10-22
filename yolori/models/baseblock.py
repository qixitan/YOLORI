# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/10/2 18:50

import torch
from torch import nn

model_sizes = {

    # darknet
    '21': [[64, 128, 256, 512, 1024], [1, 2, 2, 1]],
    '53': [[64, 128, 256, 512, 1024], [2, 8, 8, 4]],
    # cspdarknet
    "n": [[16, 32, 64, 128, 256], [1, 3, 3, 1]],
    's': [[32, 64, 128, 256, 512], [1, 3, 3, 1]],
    'm': [[48, 96, 192, 384, 768], [2, 6, 6, 2]],
    'l': [[64, 128, 256, 512, 1024], [3, 9, 9, 3]],
    'x': [[80, 160, 320, 640, 1280], [4, 12, 12, 4]],
    # resnet
    "res18": [[64, 64, 128, 256, 512], [2, 2, 2, 2]],
    "res34": [[64, 64, 128, 256, 512], [3, 4, 6, 3]],
    "res50": [[64, 64, 128, 256, 512], [3, 4, 6, 3]],
    "res101": [[64, 64, 128, 256, 512], [3, 4, 23, 3]],
    "res152": [[64, 64, 128, 256, 512], [3, 8, 36, 3]],
}


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_act(act: str = 'relu'):
    act_dic = {
        "relu": nn.ReLU(inplace=True),
        'silu': nn.SiLU(inplace=True),
        "leakyrelu": nn.LeakyReLU(inplace=True),
        "selu": nn.SELU(inplace=True),
    }
    assert act in act_dic.keys()
    return act_dic[act]


class BaseConv(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, act="relu"
    ):
        super().__init__()
        # same padding
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_act(act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int, act: str):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, kernel_size=1, stride=1, act=act
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, kernel_size=3, stride=1, act=act
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
            self, in_channels, out_channels, kernel_sizes=(5, 9, 13), act="relu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, act="relu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, kernel_size=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, act="relu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, kernel_size, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="relu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class CSPLayer(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 n=1,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False,
                 act="relu", ):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)

        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, mode="embedded_gaussian"):
        super(NonLocalBlock, self).__init__()
        assert mode in ['embedded_gaussian', 'dot_product']
        self.inplanes = in_channels
        self.mode = mode
        self.hiden_planes = in_channels//2
        self.phi_x = nn.Conv2d(self.inplanes, self.hiden_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.theta_x = nn.Conv2d(self.inplanes, self.hiden_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.g_x = nn.Conv2d(self.inplanes, self.hiden_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.out = nn.Conv2d(self.hiden_planes, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def embedded_gaussian(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= theta_x.shape[-1] ** -0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, x):
        b, _, h, w = x.size()
        phi_x = self.phi_x(x).view(b, self.hiden_planes, -1)
        theta_x = self.theta_x(x).view(b, self.hiden_planes, -1).permute(0, 2, 1).contiguous()
        g_x = self.g_x(x).view(b, self.hiden_planes, -1).permute(0, 2, 1).contiguous()

        if self.mode == 'embedded_gaussian':
            pairwise_weight = self.embedded_gaussian(theta_x, phi_x)
        else:
            pairwise_weight = self.dot_product(theta_x, phi_x)
        y = torch.matmul(pairwise_weight, g_x)
        y = y.permute(0, 2, 1).reshape(b, self.hiden_planes, h, w).contiguous()
        out = x + self.out(y)
        return out


