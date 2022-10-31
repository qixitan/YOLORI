# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/10/29

from exps.default.yolox_dior_m import Exp as yolox_Exp
from exps.default.maa_dior_l import Exp as maa_Exp
from yolori.utils import get_model_info
tsize = (224, 224)

yolox_model = yolox_Exp().get_model()
maa_model = maa_Exp().get_model()

print(get_model_info(yolox_model, tsize))
print(get_model_info(maa_model, tsize))
