#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os

from yolori.exp import MAA_Dior_Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.375
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.enable_mixup = False


