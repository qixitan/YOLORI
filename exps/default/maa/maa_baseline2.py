#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
from torch import nn
from yolori.exp import MAA_Dior_Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]


    def get_model(self):
        from yolori.models import YOLOX, NONeck2, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            backbone = NONeck2(self.depth, self.width, act=self.act)
            head = YOLOXHead(self.num_classes, self.width, strides=[16], in_channels=[512], act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

