# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/10/27

import torch
from yolori.models import YOLOPAFPN, MAA, ASFF, AS_MAA
from yolori.utils import get_model_info

depth = 1.33
width = 1.25
img_size = (224, 224)
x = torch.randn(1, 3, 224, 224)
# model = YOLOFPN(21)
# print(get_model_info(model, tsize=img_size))
model = YOLOPAFPN(depth, width)
print(get_model_info(model, tsize=img_size))
model = MAA(depth, width)
print(get_model_info(model, tsize=img_size))
model = ASFF(depth, width)
print(get_model_info(model, tsize=img_size))
model = AS_MAA(depth, width)
print(get_model_info(model, tsize=img_size))

