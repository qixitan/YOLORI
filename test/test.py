# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/10/22

import torch
from torch import nn

x = torch.randn(3, 256, 32, 32)

feature = []

feature.extend(x)
print(feature)
