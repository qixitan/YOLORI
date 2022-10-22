# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/10/22

import torch
from yolori.models import RepVGG, YOLOPAFPN, YOLOXHead
from yolori.utils import get_model_info



def func(backbone_size):
    x = torch.randn(1, 3, 224, 224)
    model = RepVGG(model_size=model_size, deploy=False)
    backbone_info = get_model_info(model, (224, 224))
    print("------backbone_info: {}, backbone_info: train: {}-------".format(backbone_size, backbone_info))

    """backbone"""
    y = model(x)
    print("backbone_out_shapes: ", y["stage3"].shape, y["stage4"].shape, y["stage5"].shape)

    """neck"""
    in_features = model.out_features
    in_channels = model.regvgg_size[model_size]["width"][-len(in_features):]
    neck = YOLOPAFPN(in_features=in_features, in_channels=in_channels)
    neck.backbone = model
    y = neck(x)
    print("neck_out_shapes: ", y[0].shape, y[1].shape, y[2].shape)

    """head"""
    head = YOLOXHead(num_classes=20, in_channels=in_channels).eval()
    y = head(y)
    print("head_out_shape: ", y.shape)


model_dic = RepVGG().regvgg_size
for model_size in model_dic.keys():
    func(model_size)



