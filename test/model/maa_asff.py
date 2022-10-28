# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/10/27

import torch
from yolori.models import AS_MAA
from yolori.utils import get_model_info

depth = 0.33
width = 0.375
x = torch.randn(1, 3, 224, 224)

model = AS_MAA(depth=depth, width=width)
y = model(x)
print(y[0].shape, y[1].shape, y[2].shape)

print(get_model_info(model, tsize=(224, 224)))
