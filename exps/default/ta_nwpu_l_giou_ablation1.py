#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from torch import nn
from yolori.exp import Nwpu_Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolori.models.backbone import CSPDarknet
            from yolori.models.neck import PAFPN
            from yolori.models.head import YOLOXHead, TOODHead
            from yolori.models import Builder
            in_channels = [256, 512, 1024]
            backbone = CSPDarknet(self.depth, self.width)
            neck = PAFPN(self.depth, self.width)
            head = TOODHead(self.num_classes, self.width, in_channels=in_channels, act=self.act, iou_type="giou", stacked_convs=1)
            self.model = Builder(backbone, neck, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
