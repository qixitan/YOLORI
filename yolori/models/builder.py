# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/12/2

import torch.nn as nn
from .backbone import CSPDarknet
from .neck import PAFPN
from .head import YOLOXHead


class Builder(nn.Module):

    def __init__(self, backbone=None, neck=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = CSPDarknet(1.0, 1.0)
        if neck is None:
            neck = PAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.neck(self.backbone(x))

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "l1_loss": l1_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs

