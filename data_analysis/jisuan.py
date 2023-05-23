# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2023/2/16

from yolori.models.losses import IOUloss
import torch

t = torch.tensor([1.5, 1, 3, 2]).unsqueeze(dim=0)
p1 = torch.tensor([3 * 3 / 4, 2 * 3 / 4, 1.5, 1]).unsqueeze(dim=0)
p2 = torch.tensor([3 * 2 / 4, 2 * 3 / 4, 1.5, 1]).unsqueeze(dim=0)
p3 = torch.tensor([3 * 2 / 4, 2 * 2 / 4, 1.5, 1]).unsqueeze(dim=0)
p4 = torch.tensor([3 * 2 / 4, 2 * 2 / 4, 1, 1.5]).unsqueeze(dim=0)

iouloss = IOUloss(loss_type="iou")
diouloss = IOUloss(loss_type="ciou")
print(iouloss(p1, t), diouloss(p1, t))
print(iouloss(p2, t), diouloss(p2, t))
print(iouloss(p3, t), diouloss(p3, t))
print(iouloss(p4, t), diouloss(p4, t))

color = [(0, 0, 0), (0, 0, 200), (0, 0, 255), (0, 200, 0), (0, 200, 200), (0, 200, 255), (0, 255, 200), (0, 255, 255),
         (200, 0, 0), (200, 0, 200), (200, 200, 0), (200, 200, 200), (200, 200, 255), (200, 255, 0), (200, 255, 200),
         (200, 255, 255), (255, 0, 255), (255, 200, 0), (255, 200, 200), (255, 200, 255), (255, 255, 255)]
