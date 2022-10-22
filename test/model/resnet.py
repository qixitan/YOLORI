# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/10/21

import torch
from yolori.models import ResNet, YOLOPAFPN, YOLOXHead


x = torch.randn(1, 3, 224, 224)
model_size = "res152"  # res34, res50, res101, res152

model = ResNet(model_size=model_size)
in_features = model.out_features
in_channels = model.resnet_sizes[model_size][0][-len(in_features):]
neck = YOLOPAFPN(in_features=in_features, in_channels=in_channels)
neck.backbone = model
y = neck(x)
print(y[0].shape, y[1].shape, y[2].shape)
head = YOLOXHead(num_classes=20, in_channels=in_channels).eval()
y = head(y)
print(y.shape)

