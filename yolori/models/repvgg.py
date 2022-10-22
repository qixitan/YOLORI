# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/10/22

import torch.nn as nn
from .network_blocks import RepVGGBlock


class RepVGG(nn.Module):
    def __init__(self, num_blocks, width=None, override_groups_map=None, deploy=False, use_se=False,
                 out_features=('stage3', 'stage4', 'stage5'), act="relu"):
        super(RepVGG, self).__init__()
        assert len(width == 4)
        self.deploy = deploy
        self.override_groups_map = override_groups_map
        assert 0 not in self.override_groups_map
        self.use_se = use_se
        self.act = act
        self.out_features = out_features

        self.in_channels = min(64, int(64 * width[0]))
        self.stage1 = RepVGGBlock(in_channels=3, out_channels=self.in_channels, kernel_size=3, stride=2, padding=1,
                                  deploy=self.deploy, use_se=self.use_se)
        self.cur_layer_idx = 1
        self.stage2 = self._make_stage(int(64 * width[0]), num_blocks[0], stride=2)
        self.stage3 = self._make_stage(int(128 * width[1]), num_blocks[1], stride=2)
        self.stage4 = self._make_stage(int(256 * width[2]), num_blocks[2], stride=2)
        self.stage5 = self._make_stage(int(512 * width[3]), num_blocks[3], stride=2)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []

        for stride in strides:
            cur_group = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(
                in_channels=self.in_channels, out_channels=planes, kernel_size=3, stride=stride, padding=1,
                groups=cur_group, deploy=self.deploy, use_se=self.use_se, act=self.act
            ))
            self.in_channels = planes
            self.cur_layer_idx += 1
        return nn.ModuleList(blocks)

    def forward(self, x):
        outputs = {}
        out = self.stage1(x)
        outputs["stage1"] = out
        out = self.stage2(out)
        outputs["stage2"] = out
        out = self.stage3(out)
        outputs["stage3"] = out
        out = self.stage4(out)
        outputs["stage4"] = out
        out = self.stage5(out)
        outputs["stage5"] = out
        return {k: v for k, v in outputs.items() if k in self.out_features}

optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}

