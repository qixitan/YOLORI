# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/10/6 21:14

from torch import nn
from .network_blocks import get_activation


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, act="relu"):
        super(BasicBlock, self).__init__()
        self.act = get_activation(act)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = self.act(out)
        return out


class Bottleneck(nn.Module):
    # 在resnet50、101、152中使用的残差块
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, act="relu"):
        super(Bottleneck, self).__init__()
        self.act = get_activation(act)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels//self.expansion, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels//self.expansion)

        self.conv2 = nn.Conv2d(
            out_channels//self.expansion, out_channels//self.expansion, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels//self.expansion)

        self.conv3 = nn.Conv2d(
            out_channels//self.expansion, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = self.act(out)
        return out


class ResNet(nn.Module):
    resnet_sizes = {
        "res18": [[64, 64, 128, 256, 512], [2, 2, 2, 2]],
        "res34": [[64, 64, 128, 256, 512], [3, 4, 6, 3]],
        "res50": [[64, 256, 512, 1024, 2048], [3, 4, 6, 3]],
        "res101": [[64, 256, 512, 1024, 2048], [3, 4, 23, 3]],
        "res152": [[64, 256, 512, 1024, 2048], [3, 8, 36, 3]],
    }

    def __init__(self, model_size="res18", out_features=('stage3', 'stage4','stage5'),in_channels=3, act="relu"):
        super(ResNet, self).__init__()
        assert model_size in self.resnet_sizes.keys(), "please select correct model_size from {}".format(
            self.resnet_sizes.keys())
        self.out_features = out_features
        layer_out_channels, layer_num_blocks = self.resnet_sizes[model_size]
        self.act = act
        self.in_channels = layer_out_channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, layer_out_channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(layer_out_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        block = BasicBlock if model_size in ("res18", "res34") else Bottleneck
        self.layer1 = self._make_layer(
            block, layer_out_channels[1], layer_num_blocks[0], stride=1
        )
        self.layer2 = self._make_layer(
            block, layer_out_channels[2], layer_num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, layer_out_channels[3], layer_num_blocks[2], stride=2
        )
        self.layer4 = self._make_layer(
            block, layer_out_channels[4], layer_num_blocks[3], stride=2
        )
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.linear = nn.Linear(layer_out_channels[4], 1000)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        # print(strides)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, act=self.act))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.layer1(x)
        outputs['stage2'] = x
        x = self.layer2(x)
        outputs['stage3'] = x
        x = self.layer3(x)
        outputs['stage4'] = x
        x = self.layer4(x)
        outputs['stage5'] = x
        # x = self.avgpool(x)     # used for classification， useless for detector
        # out = x.view(x.size(0), -1)
        # out = self.linear(out)
        return {k: v for k, v in outputs.items() if k in self.out_features}

