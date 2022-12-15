# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/12/2

# only neck
from .asff import ASFF
from .fpn import FPN
from .f_neck import (
    CBAM_PAFPN, CBAM_PANFPN_SGNonlocalAttention, PAFPN_ASFF, PANFPN_SGNonlocalAttention
)
from .maa import MAA
from .noneck import NoNeck, NoNeck1, NoNeck2
from .pafpn import PAFPN
from .sga import (CGNonlocalAttention, SGNonlocalAttention)

__all__ = [
    "FPN", "MAA", "PAFPN", "ASFF", "PAFPN_ASFF", "CBAM_PANFPN_SGNonlocalAttention", "CBAM_PAFPN",
    "PANFPN_SGNonlocalAttention", "SGNonlocalAttention", "CGNonlocalAttention", "NoNeck", "NoNeck1", "NoNeck2"
]
