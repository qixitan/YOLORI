# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/10/22

import torch.nn as nn
from .network_blocks import RepVGGBlock

optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


class RepVGG(nn.Module):
    regvgg_size = {
        "RepVGG_A0": {"blocks": [2, 4, 14, 1], "width": [48, 96, 192, 1280],
                      "override_groups_map": None, "deploy": False, "use_se": False},
        "RepVGG_A1": {"blocks": [2, 4, 14, 1], "width": [64, 128, 256, 1280],
                      "override_groups_map": None, "deploy": False, "use_se": False},
        "RepVGG_A2": {"blocks": [2, 4, 14, 1], "width": [96, 192, 384, 1408],
                      "override_groups_map": None, "deploy": False, "use_se": False},

        "RepVGG_B0": {"blocks": [4, 6, 16, 1], "width": [64, 128, 256, 1280],
                      "override_groups_map": None, "deploy": False, "use_se": False},
        "RepVGG_B1": {"blocks": [4, 6, 16, 1], "width": [124, 256, 512, 2048],
                      "override_groups_map": None, "deploy": False, "use_se": False},

        "RepVGG_B1g2": {"blocks": [4, 6, 16, 1], "width": [124, 256, 512, 2048],
                      "override_groups_map": g2_map, "deploy": False, "use_se": False},
        "RepVGG_B1g4": {"blocks": [4, 6, 16, 1], "width": [124, 256, 512, 2048],
                        "override_groups_map": g4_map, "deploy": False, "use_se": False},

        "RepVGG_B2": {"blocks": [4, 6, 16, 1], "width": [160, 320, 640, 2560],
                      "override_groups_map": None, "deploy": False, "use_se": False},
        "RepVGG_B2g2": {"blocks": [4, 6, 16, 1], "width": [160, 320, 640, 2560],
                      "override_groups_map": g2_map, "deploy": False, "use_se": False},
        "RepVGG_B2g4": {"blocks": [4, 6, 16, 1], "width": [160, 320, 640, 2560],
                        "override_groups_map": g4_map, "deploy": False, "use_se": False},

        "RepVGG_B3": {"blocks": [4, 6, 16, 1], "width": [192, 384, 768, 2560],
                      "override_groups_map": None, "deploy": False, "use_se": False},
        "RepVGG_B3g2": {"blocks": [4, 6, 16, 1], "width": [192, 384, 768, 2560],
                      "override_groups_map": g2_map, "deploy": False, "use_se": False},
        "RepVGG_B3g4": {"blocks": [4, 6, 16, 1], "width": [192, 384, 768, 2560],
                        "override_groups_map": g4_map, "deploy": False, "use_se": False},

        "RepVGG_D2se": {"blocks": [8, 14, 24, 1], "width": [160, 320, 640, 2560],
                        "override_groups_map": None, "deploy": False, "use_se": True},
    }

    def __init__(self, model_size="RepVGG_A0", deploy=False, out_features=('stage3', 'stage4', 'stage5'), act="relu"):
        super(RepVGG, self).__init__()
        assert model_size in self.regvgg_size.keys(), "please select correct model_size from {}".format(
            self.regvgg_size.keys())
        model = self.regvgg_size[model_size]
        width = model["width"]
        num_blocks = model["blocks"]
        if deploy is not False:
            self.deploy = deploy
        else:
            self.deploy = model["deploy"]
        self.use_se = model["use_se"]
        self.override_groups_map = model["override_groups_map"] or dict()
        assert 0 not in self.override_groups_map
        assert len(width) == 4

        self.act = act
        self.out_features = out_features

        self.in_channels = min(64, int(width[0]))
        self.stage1 = RepVGGBlock(in_channels=3, out_channels=self.in_channels, kernel_size=3, stride=2, padding=1,
                                  deploy=self.deploy, use_se=self.use_se)
        self.cur_layer_idx = 1
        self.stage2 = self._make_stage(int(width[0]), num_blocks[0], stride=2)
        self.stage3 = self._make_stage(int(width[1]), num_blocks[1], stride=2)
        self.stage4 = self._make_stage(int(width[2]), num_blocks[2], stride=2)
        self.stage5 = self._make_stage(int(width[3]), num_blocks[3], stride=2)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_channels, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy,
                                      use_se=self.use_se, act=self.act))
            self.in_channels = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

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



