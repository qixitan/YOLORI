# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/10/24

import torch
from yolori.models import ASFF, YOLOX, YOLOXHead
from yolori.utils import get_model_info

depth = 0.33
width = 0.375
# x = torch.randn(1, 3, 224, 224)
x = torch.randn(1, 3, 256, 256)

# model = ASFF(depth=depth, width=width)
#
# y = model(x)
# print(y[0].shape, y[1].shape, y[2].shape)

model = YOLOX(ASFF(depth=depth, width=width), YOLOXHead(num_classes=20, width=width))
# targ = torch.Tensor([[[1, 0.1, 0.1, 0.1, 0.1],
#                         [2, 0.1, 0.1, 0.1, 0.2],
#                         [1, 0.1, 0.1, 0.5, 0.1]]])

labels = torch.Tensor([[[1, 0.1, 0.1, 0.1, 0.1],
                        [2, 0.1, 0.1, 0.1, 0.2],
                        [1, 0.1, 0.1, 0.5, 0.1]]])
y = model(x, labels)
print(y)


