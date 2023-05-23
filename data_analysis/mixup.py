# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2023/2/16

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib._color_data as mcd
from yolori.data import (
    DIORDetection,
    TrainTransform,
    YoloBatchSampler,
    DataLoader,
    InfiniteSampler,
    MosaicDetection,
    worker_init_reset_seed,
    DIORDataset,
)

dataset = DIORDetection(
    data_dir="/Home/guest/Datasets/DIORdevkit",
    image_sets=[("2018", "train"), ("2018", "val")],
    img_size=(800, 800),
    preproc=TrainTransform(max_labels=50, flip_prob=0, hsv_prob=0),
    cache=False
)

color = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond',
         'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
         'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey',
         'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
         'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink',
         'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick'][:20]

t1 = dataset[1]
img, label = t1[0], t1[1]
img = img.transpose(1, 2, 0).astype("int")
plt.show()
