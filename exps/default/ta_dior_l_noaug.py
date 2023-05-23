#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from torch import nn
from yolori.exp import Dior_Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.375
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.random_size = (14, 26)
        self.no_aug_epochs = -1
        self.max_epoch = 40

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolori.models.backbone import CSPDarknet, Cascade_CSPDarknet
            from yolori.models.neck import PAFPN
            from yolori.models.head import YOLOXHead, TAHead
            from yolori.models import Builder
            in_channels = [256, 512, 1024]
            backbone = Cascade_CSPDarknet(dep_mul=self.depth, wid_mul=self.width, layertype="attention")
            neck = PAFPN(self.depth, self.width)
            head = TAHead(self.num_classes, self.width, in_channels=in_channels, act=self.act, iou_type="ciou")
            self.model = Builder(backbone, neck, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
