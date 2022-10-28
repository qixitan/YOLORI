# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/10/27

import torch
from yolori.models import CSPDarknet, ResNet, RepVGG
from yolori.utils import get_model_info

depth = 1.33
width = 1.25
img_size = (224, 224)
x = torch.randn(1, 3, 224, 224)
model = CSPDarknet(dep_mul=depth, wid_mul=width)
print(get_model_info(model, tsize=img_size))
model = ResNet(model_size="res152")
print(get_model_info(model, tsize=img_size))
model = RepVGG(model_size="RepVGG_B0")
print(get_model_info(model, tsize=img_size))
