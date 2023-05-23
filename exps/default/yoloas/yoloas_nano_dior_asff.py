# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/11/17

import os
from torch import nn
from yolori.exp import Dior_Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (416, 416)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.test_size = (416, 416)
        self.mosaic_prob = 0.5
        self.enable_mixup = False
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            # from yolori.models import YOLOX, YOLOPAFPN, YOLOXHead, CBAM_YOLOPAFPN
            from yolori.models.backbone import CSPDarknet
            from yolori.models.neck import PAFPN_ASFF
            from yolori.models.head import YOLOXHead
            from yolori.models import Builder
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            # backbone = CBAM_YOLOPAFPN(
            #     self.depth, self.width, in_channels=in_channels,
            #     act=self.act, depthwise=True,
            # )
            # head = YOLOXHead(
            #     self.num_classes, self.width, in_channels=in_channels,
            #     act=self.act, depthwise=True
            # )
            # self.model = YOLOX(backbone, head)
            backbone = CSPDarknet(self.depth, self.width, act=self.act, depthwise=True)
            neck = PAFPN_ASFF(self.depth, self.width, depthwise=True)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act, depthwise=True)
            self.model = Builder(backbone, neck, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model