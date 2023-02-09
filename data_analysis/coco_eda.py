# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2023/1/31
# reference resources: https://github.com/gy-7/EDA
import os
import seaborn as sns
import pycocotools.coco
import matplotlib.pyplot as plt

root_dir = os.getcwd()
train_ann_fp = os.path.join(root_dir, 'annotations', 'instances_train2017.json')
val_ann_fp = os.path.join(root_dir, 'annotations', 'instances_val2017.json')


class COCO_EDA:
    def __init__(self, json_file, type='train'):
        self.COCO_SMALL_SCALE = 32
        self.COCO_MEDIUM_SCALE = 96

        self.json_file = json_file
        coco = pycocotools.coco.COCO(json_file)

        self.type = type
        self.imgs = coco.dataset['images']
        self.anns = coco.dataset['annotations']
        self.cats = coco.dataset['categories']
        self.img_ids = coco.getImgIds()
        self.ann_ids = coco.getAnnIds()
        self.cat_ids = coco.getCatIds()

        self.cat2imgs = coco.catToImgs
        self.img2anns = coco.imgToAnns

        self.imgs_num = len(self.imgs)
        self.objs_num = len(self.anns)

        # data to be collected
        self.small_objs_num = 0
        self.medium_objss_num = 0
        self.large_objss_num = 0

        self.small_objs = []
        self.medium_objs = []
        self.large_objs = []

        self.cat2objs = {}
        self.small_cat2objs = {}  # small objects classes distribution
        self.medium_cat2objs = {}  # medium objects classes distribution
        self.large_cat2objs = {}  # large objects classes distribution
        self.cat2objs_num = {}  # objects classes distribution
        self.small_cat2objs_num = {}  # small objects classes distribution
        self.medium_cat2objs_num = {}  # medium objects classes distribution
        self.large_cat2objs_num = {}  # large objects classes distribution

        # plot use data
        self.catid2name = {}  # 用于绘图中显示类别名字
        self.cats_plot = []  # coco 所有尺寸目标的类别分布
        self.small_cats_plot = []  # 小目标中每个类的分布情况
        self.medium_cats_plot = []  # 中目标中每个类的分布情况
        self.large_cats_plot = []  # 大目标中每个类的分布情况

        # 每个类的小，中，大目标的数量
        self.size_distribution = {}


def collect_data(coco):
    # collect small, medium, large objects
    for ann in coco.anns:
        if ann['area'] < coco.COCO_SMALL_SCALE ** 2:
            coco.small_objs_num += 1
            coco.small_objs.append(ann)
        elif ann['area'] < coco.COCO_MEDIUM_SCALE ** 2:
            coco.medium_objs.append(ann)
            coco.medium_objss_num += 1
        else:
            coco.large_objs.append(ann)
            coco.large_objss_num += 1

    for i in coco.cat_ids:
        coco.cat2objs[i] = []
        coco.small_cat2objs[i] = []
        coco.medium_cat2objs[i] = []
        coco.large_cat2objs[i] = []
        coco.cat2objs_num[i] = 0
        coco.small_cat2objs_num[i] = 0
        coco.medium_cat2objs_num[i] = 0
        coco.large_cat2objs_num[i] = 0
        coco.size_distribution[i] = []

    for i in coco.cats:
        coco.catid2name[i['id']] = i['name']

    # collect small, medium, large class distribution
    for i in coco.anns:
        coco.cat2objs[i['category_id']].append(i)
        coco.cat2objs_num[i['category_id']] += 1
        coco.cats_plot.append(coco.catid2name[i['category_id']])
        if i['area'] < coco.COCO_SMALL_SCALE ** 2:
            coco.small_cat2objs[i['category_id']].append(i)
            coco.small_cat2objs_num[i['category_id']] += 1
            coco.small_cats_plot.append(coco.catid2name[i['category_id']])
            coco.size_distribution[i['category_id']].append('s')
        elif i['area'] < coco.COCO_MEDIUM_SCALE ** 2:
            coco.medium_cat2objs[i['category_id']].append(i)
            coco.medium_cat2objs_num[i['category_id']] += 1
            coco.medium_cats_plot.append(coco.catid2name[i['category_id']])
            coco.size_distribution[i['category_id']].append('m')
        else:
            coco.large_cat2objs[i['category_id']].append(i)
            coco.large_cat2objs_num[i['category_id']] += 1
            coco.large_cats_plot.append(coco.catid2name[i['category_id']])
            coco.size_distribution[i['category_id']].append('l')

    assert len(coco.small_objs) == coco.small_objs_num == sum(coco.small_cat2objs_num.values())
    assert len(coco.medium_objs) == coco.medium_objss_num == sum(coco.medium_cat2objs_num.values())
    assert len(coco.large_objs) == coco.large_objss_num == sum(coco.large_cat2objs_num.values())
    assert len(coco.anns) == coco.objs_num == sum(coco.cat2objs_num.values())


def plot_coco_class_distribution(plot_data, plot_order, save_fp, plot_title, plot_y_heigh,
                                 plot_y_heigh_residual=[1800, 100]):
    # 绘制coco数据集的类别分布
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 8))  # 图片的宽和高，单位为inch
    plt.title(plot_title, fontsize=9)  # 标题
    plt.xlabel('class', fontsize=8)  # x轴名称
    plt.ylabel('counts', fontsize=8)  # y轴名称
    plt.xticks(rotation=90, fontsize=8)  # x轴标签竖着显示
    plt.yticks(fontsize=8)
    for x, y in enumerate(plot_y_heigh):
        if 'train' in save_fp:
            plt.text(x, y + plot_y_heigh_residual[0], '%s' % y, ha='center', fontsize=7, rotation=90)
        else:
            plt.text(x, y + plot_y_heigh_residual[1], '%s' % y, ha='center', fontsize=7, rotation=90)
    ax = sns.countplot(x=plot_data, palette="PuBu_r", order=plot_order)  # 绘制直方图，palette调色板，蓝色由浅到深渐变。
    # palette样式：https://blog.csdn.net/panlb1990/article/details/103851983
    plt.savefig(os.path.join(save_fp), dpi=500)
    plt.show()


def plot_size_distribution(plot_data, save_fp, plot_title, plot_order=['s', 'm', 'l']):
    sns.set_style("whitegrid")
    plt.figure(figsize=(21, 35))  # 图片的宽和高，单位为inch
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=1, hspace=1.5)  # 调整子图间距

    for idx, size_data in enumerate(plot_data.values()):
        plt.subplot(10, 8, idx + 1)
        plt.xticks(rotation=0, fontsize=18)  # x轴标签竖着显示
        plt.yticks(fontsize=18)
        plt.xlabel('size', fontsize=20)  # x轴名称
        plt.ylabel('count', fontsize=20)  # y轴名称
        plt.title(plot_title[idx], fontsize=24)  # 标题
        sns.countplot(x=size_data, palette="PuBu_r", order=plot_order)  # 绘制直方图，palette调色板，蓝色由浅到深渐变。

    plt.savefig(save_fp, dpi=500, pad_inches=0)
    plt.show()


def run_plot_coco_class_distribution(coco, save_dir):
    # # 绘制coco数据集的类别分布
    plot_order = [i for i in coco.catid2name.values()]

    plot_heigh = [i for i in coco.cat2objs_num.values()]
    save_fp = os.path.join(save_dir, f'coco_{coco.type}_class_distribution.png')
    plot_coco_class_distribution(coco.cats_plot, plot_order, save_fp, 'COCO train2017 class distribution', plot_heigh,
                                 plot_y_heigh_residual=[1800, 100])

    plot_heigh = [i for i in coco.small_cat2objs_num.values()]
    save_fp = os.path.join(save_dir, f'coco_{coco.type}_small_class_distribution.png')
    plot_coco_class_distribution(coco.small_cats_plot, plot_order, save_fp, 'COCO train2017 small class distribution',
                                 plot_heigh,
                                 plot_y_heigh_residual=[900, 50])

    plot_heigh = [i for i in coco.medium_cat2objs_num.values()]
    save_fp = os.path.join(save_dir, f'coco_{coco.type}_medium_class_distribution.png')
    plot_coco_class_distribution(coco.medium_cats_plot, plot_order, save_fp, 'COCO train2017 medium class distribution',
                                 plot_heigh, plot_y_heigh_residual=[900, 50])

    plot_heigh = [i for i in coco.large_cat2objs_num.values()]
    save_fp = os.path.join(save_dir, f'coco_{coco.type}_large_class_distribution.png')
    plot_coco_class_distribution(coco.large_cats_plot, plot_order, save_fp, 'COCO train2017 large class distribution',
                                 plot_heigh,
                                 plot_y_heigh_residual=[900, 50])


def run_plot_coco_size_distribution(coco, save_dir):
    # 绘制coco数据集各类别的尺寸分布
    plot_order = [i for i in coco.catid2name.values()]
    save_fp = os.path.join(save_dir, f'coco_{coco.type}_size_distribution.png')
    plot_size_distribution(coco.size_distribution, save_fp, plot_order)


if __name__ == '__main__':
    cwd = os.getcwd()
    coco_res_dir = os.path.join(cwd, "coco2017_results")
    if not os.path.exists(coco_res_dir):
        os.makedirs(coco_res_dir)

    print("analyze coco train dataset...")
    print("-" * 50)
    coco_train = COCO_EDA(train_ann_fp, type='train')
    collect_data(coco_train)
    print("coco train images num:", coco_train.imgs_num)
    print("coco train objects num:", coco_train.objs_num)
    print("coco small objects num:", coco_train.small_objs_num)
    print("coco medium objects num:", coco_train.medium_objss_num)
    print("coco large objects num:", coco_train.large_objss_num)
    print("coco small objects percent:", coco_train.small_objs_num / coco_train.objs_num)
    print("coco medium objects percent:", coco_train.medium_objss_num / coco_train.objs_num)
    print("coco large objects percent:", coco_train.large_objss_num / coco_train.objs_num)
    run_plot_coco_class_distribution(coco_train, coco_res_dir)
    run_plot_coco_size_distribution(coco_train, coco_res_dir)
    print("-" * 50)
    print()

    print("analyze coco val dataset...")
    print("-" * 50)
    coco_val = COCO_EDA(val_ann_fp, type='val')
    collect_data(coco_val)
    print("coco val images num:", coco_val.imgs_num)
    print("coco val objects num:", coco_val.objs_num)
    print("coco small objects num:", coco_val.small_objs_num)
    print("coco medium objects num:", coco_val.medium_objss_num)
    print("coco large objects num:", coco_val.large_objss_num)
    print("coco small objects percent:", coco_val.small_objs_num / coco_val.objs_num)
    print("coco medium objects percent:", coco_val.medium_objss_num / coco_val.objs_num)
    print("coco large objects percent:", coco_val.large_objss_num / coco_val.objs_num)
    run_plot_coco_class_distribution(coco_val, coco_res_dir)
    run_plot_coco_size_distribution(coco_val, coco_res_dir)
    print("-" * 50)