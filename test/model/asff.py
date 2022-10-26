# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/10/24

import torch
from yolori.models import ASFF, YOLOX, YOLOXHead
from yolori.utils import get_model_info

depth = 0.33
width = 0.375
x = torch.randn(1, 3, 224, 224)

model = ASFF(depth=depth, width=width)

y = model(x)
print(y[0].shape, y[1].shape, y[2].shape)

model = YOLOX(model, YOLOXHead(num_classes=20, width=width))
print(get_model_info(model, tsize=(800, 800)))
