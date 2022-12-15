# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/10/24

import torch
from torch import nn
import torch.nn.functional as F
from yolori.models.network_blocks import BaseConv


class ASFF_Block2(nn.Module):
    def __init__(self, level, multiplier=1.0, rfb=False, vis=False):
        """
        multiplier should be 1, 0.5
        which means, the channel of ASFF_Block2 can be
        512, 256, 128 -> multiplier=0.5
        1024, 512, 256 -> multiplier=1
        For even smaller, you need change code manually.
        """
        super().__init__()
        self.level = level
        self.dim = [int(1024 * multiplier), int(512 * multiplier),
                    int(256 * multiplier)]

        self.inter_dim = self.dim[abs(self.level - 5)]
        if level == 5:
            # self.stride_level_1 = BaseConv(int(512 * multiplier), self.inter_dim, 3, 2)
            # self.stride_level_2 = BaseConv(int(256 * multiplier), self.inter_dim, 3, 2)
            self.stride_attention = BaseConv(int(512 * multiplier), self.inter_dim, 3, 2)
            # self.expand = BaseConv(self.inter_dim, int(1024 * multiplier), 3, 1)
            self.expand = BaseConv(self.inter_dim, int(1024 * multiplier), 3, 1)

        elif level == 4:
            # self.compress_level_0 = BaseConv(int(1024 * multiplier), self.inter_dim, 1, 1)
            # self.stride_level_2 = BaseConv(int(256 * multiplier), self.inter_dim, 3, 2)
            # self.expand = BaseConv(self.inter_dim, int(512 * multiplier), 3, 1)
            self.expand = BaseConv(self.inter_dim, int(512 * multiplier), 3, 1)
        elif level == 3:
            # self.compress_level_0 = BaseConv(int(1024 * multiplier), self.inter_dim, 1, 1)
            # self.compress_level_1 = BaseConv(int(512 * multiplier), self.inter_dim, 1, 1)
            self.compress_attention = BaseConv(int(512 * multiplier), self.inter_dim, 1, 1)
            # self.expand = BaseConv(self.inter_dim, int(256 * multiplier), 3, 1)
            self.expand = BaseConv(self.inter_dim, int(256 * multiplier), 3, 1)

        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16
        # self.weight_level_0 = BaseConv(self.inter_dim, compress_c, 1, 1)
        # self.weight_level_1 = BaseConv(self.inter_dim, compress_c, 1, 1)
        # self.weight_level_2 = BaseConv(self.inter_dim, compress_c, 1, 1)
        self.weight_attention = BaseConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level = BaseConv(self.inter_dim, compress_c, 1, 1)

        # self.weight_levels = BaseConv(compress_c * 3, 3, 1, 1)
        self.weight_levels = BaseConv(compress_c * 2, 3, 1, 1)
        self.vis = vis

    def forward(self, x):
        """
        512, x[256, 512, 1024]
        from small -> large
        """
        # x_level_0 = x[2]  # 1024
        # x_level_1 = x[1]  # 512
        # x_level_2 = x[0]  # 256

        x_attention = x[0]  # 512
        x_level = x[1]  # ? [256, 512, 1024]
        if self.level == 5:
            # level_0_resized = x_level_0
            # level_1_resized = self.stride_level_1(x_level_1)
            # level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            # level_2_resized = self.stride_level_2(level_2_downsampled_inter)

            level_attention = self.stride_attention(x_attention)

        elif self.level == 4:
            # level_0_compressed = self.compress_level_0(x_level_0)
            # level_0_resized = F.interpolate(
            #     level_0_compressed, scale_factor=2, mode='nearest')
            # level_1_resized = x_level_1
            # level_2_resized = self.stride_level_2(x_level_2)
            level_attention = x_attention
        elif self.level == 3:
            # level_0_compressed = self.compress_level_0(x_level_0)
            # level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            # x_level_1_compressed = self.compress_level_1(x_level_1)
            # level_1_resized = F.interpolate(x_level_1_compressed, scale_factor=2, mode='nearest')
            # level_2_resized = x_level_2
            level_attention = self.compress_attention(F.interpolate(x_attention, scale_factor=2, mode="nearest"))

        # level_0_weight_v = self.weight_level_0(level_0_resized)
        # level_1_weight_v = self.weight_level_1(level_1_resized)
        # level_2_weight_v = self.weight_level_2(level_2_resized)

        attention_weight_v = self.weight_attention(level_attention)
        level_weight_v = self.weight_level(x_level)

        # levels_weight_v = torch.cat( (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight_v = torch.cat([attention_weight_v, level_weight_v], 1)
        # levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = self.weight_levels(levels_weight_v)
        # levels_weight = F.softmax(levels_weight, dim=1)
        levels_weight = F.softmax(levels_weight, dim=1)

        # fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
        #                     level_1_resized * levels_weight[:, 1:2, :, :] + \
        #                     level_2_resized * levels_weight[:, 2:, :, :]

        fused_out_reduced = level_attention * levels_weight[:, 0:1, :, :] + \
                            x_level * levels_weight[:, 1:2, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class ASFF_Block3(nn.Module):
    def __init__(self, level, multiplier=1.0, rfb=False, vis=False):
        """
        multiplier should be 1, 0.5
        which means, the channel of ASFF_Block3 can be
        512, 256, 128 -> multiplier=0.5
        1024, 512, 256 -> multiplier=1
        For even smaller, you need change code manually.
        """
        super().__init__()
        self.level = level
        self.dim = [int(1024 * multiplier), int(512 * multiplier),
                    int(256 * multiplier)]
        # print(self.dim)

        self.inter_dim = self.dim[abs(self.level-5)]
        if level == 5:
            self.stride_level_1 = BaseConv(int(512 * multiplier), self.inter_dim, 3, 2)

            self.stride_level_2 = BaseConv(int(256 * multiplier), self.inter_dim, 3, 2)

            self.expand = BaseConv(self.inter_dim, int(
                1024 * multiplier), 3, 1)
        elif level == 4:
            self.compress_level_0 = BaseConv(
                int(1024 * multiplier), self.inter_dim, 1, 1)
            self.stride_level_2 = BaseConv(
                int(256 * multiplier), self.inter_dim, 3, 2)
            self.expand = BaseConv(self.inter_dim, int(512 * multiplier), 3, 1)
        elif level == 3:
            self.compress_level_0 = BaseConv(
                int(1024 * multiplier), self.inter_dim, 1, 1)
            self.compress_level_1 = BaseConv(
                int(512 * multiplier), self.inter_dim, 1, 1)
            self.expand = BaseConv(self.inter_dim, int(
                256 * multiplier), 3, 1)

        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16
        self.weight_level_0 = BaseConv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = BaseConv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BaseConv(
            self.inter_dim, compress_c, 1, 1)

        self.weight_levels = BaseConv(
            compress_c * 3, 3, 1, 1)
        self.vis = vis

    def forward(self, x):
        """
        #
        256, 512, 1024
        from small -> large
        """
        x_level_0 = x[2]  # 1024
        x_level_1 = x[1]  # 512
        x_level_2 = x[0]  # 256

        if self.level == 5:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(
                x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 4:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 3:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(
                x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class ASFF(nn.Module):
    def __init__(self, width=1.0, in_features=("dark3", "dark4", "dark5")):
        super().__init__()
        self.in_features = in_features
        self.asff3 = ASFF_Block3(level=3, multiplier=width)
        self.asff4 = ASFF_Block3(level=4, multiplier=width)
        self.asff5 = ASFF_Block3(level=5, multiplier=width)

    def forward(self, input):
        features = [input[f] for f in self.in_features]
        asff_out3 = self.asff3(features)
        asff_out4 = self.asff4(features)
        asff_out5 = self.asff5(features)
        out = (asff_out3, asff_out4, asff_out5)
        return out
