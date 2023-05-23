#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from torch import nn

from .dior_base import Dior_Exp as BaseExp


class MAA_Dior_Exp(BaseExp):
    def __init__(self):
        super(MAA_Dior_Exp, self).__init__()

    def get_model(self):
        from yolori.models import YOLOX, MAA, YOLOXHeadN

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = MAA(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHeadN(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model


