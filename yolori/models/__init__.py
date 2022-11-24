# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/10/2 18:50

from .darknet import Darknet, CSPDarknet
from .losses import IOUloss
from .yolo_fpn import YOLOFPN
from .yolo_head import YOLOXHead, YOLOXHeadN, YOLOXHeadN_
from .yolo_pafpn import YOLOPAFPN
from .yolox import YOLOX
# from .network_blocks import *


# add net
# backbone
from .resnet import ResNet
from .repvgg import RepVGG
from .yoloas import AS_CSPDarknet, SE_CSPDarknet, CBAM_CSPDarknet

# neck
from .maa import MAA
from .asff import ASFF
from .as_maa import AS_MAA
from .noneck import NONeck, NONeck1, NONeck2
from .yoloas import SE_YOLOPAFPN, CBAM_YOLOPAFPN, KSE_YOLOPAFPN

# from .yolo_pafpn import YOLOPAFPN_Space
# head
from .yolo_head import CoupleHead

