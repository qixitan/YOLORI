# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/12/1
# reference from https://github.com/alibaba/EasyCV


import torch
from torch import nn
import torch.nn.functional as F
from yolori.models.network_blocks import normal_init, BaseConv
from .yolox_head import YOLOXHead


class TaskDecomposition(nn.Module):
    """Task decomposition module in task-aligned predictor of TOOD.

    Args:
        feat_channels (int): Number of feature channels in TOOD head.
        stacked_convs (int): Number of conv layers in TOOD head.
        la_down_rate (int): Downsample rate of layer attention.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
    """

    def __init__(self,
                 feat_channels,
                 stacked_convs=6,
                 la_down_rate=8,
                 # conv_cfg=None,
                 # norm_cfg=None
                 ):
        super(TaskDecomposition, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        # self.norm_cfg = norm_cfg
        self.layer_attention = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels // la_down_rate, self.stacked_convs, 1, padding=0), nn.Sigmoid()
        )

        # self.reduction_conv = ConvModule(
        #     self.in_channels, self.feat_channels, 1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
        #     bias=norm_cfg is None
        # )
        self.reduction_conv = BaseConv(
            in_channels=self.in_channels, out_channels=self.feat_channels, kernel_size=1, stride=1, act="relu")

    def init_weights(self):
        for m in self.layer_attention.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.reduction_conv.conv, std=0.01)

    def forward(self, feat, avg_feat=None):
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.layer_attention(avg_feat)

        # here we first compute the product between layer attention weight and
        # conv weight, and then compute the convolution between new conv weight
        # and feature map, in order to save memory and FLOPs.
        conv_weight = weight.reshape(
            b, 1, self.stacked_convs,
            1) * self.reduction_conv.conv.weight.reshape(
            1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels,
                                          self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h,
                                                    w)
        if self.reduction_conv.bn is not None:
            feat = self.reduction_conv.bn(feat)
        feat = self.reduction_conv.act(feat)

        return feat


class TOODHead(YOLOXHead):
    def __init__(self,
                 num_classes,
                 width=1.0,
                 strides=[8, 16, 32],
                 in_channels=[256, 512, 1024],
                 act="silu",
                 depthwise=False,
                 stacked_convs=3,
                 la_down_rate=32):
        super(TOODHead, self).__init__(
            num_classes=num_classes, width=width, strides=strides, in_channels=in_channels, act=act,
            depthwise=depthwise, )
        self.width = width
        self.stacked_convs = stacked_convs
        self.feat_channels = int(256 * self.width)
        self.cls_decomps = nn.ModuleList()
        self.reg_decomps = nn.ModuleList()

        self.inter_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.cls_decomps.append(
                TaskDecomposition(self.feat_channels, self.stacked_convs,
                                  self.stacked_convs * la_down_rate,))
            self.reg_decomps.append(
                TaskDecomposition(self.feat_channels, self.stacked_convs,
                                  self.stacked_convs * la_down_rate,))
        for i in range(self.stacked_convs):
            chn = self.feat_channels

            self.inter_convs.append(
                BaseConv(chn, self.feat_channels, 3,stride=1))

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_decomp, reg_decomp, cls_conv, reg_conv, stride_this_level,
                x) in enumerate(
                    zip(self.cls_decomps, self.reg_decomps, self.cls_convs,
                        self.reg_convs, self.strides, xin)):
            x = self.stems[k](x)

            inter_feats = []
            for inter_conv in self.inter_convs:
                x = inter_conv(x)
                inter_feats.append(x)
            feat = torch.cat(inter_feats, 1)

            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_x = cls_decomp(feat, avg_feat)
            reg_x = reg_decomp(feat, avg_feat)

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type())
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(
                        1, grid.shape[1]).fill_(stride_this_level).type_as(
                            xin[0]))
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(batch_size, self.n_anchors, 4,
                                                 hsize, wsize)
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4)
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat([
                    reg_output,
                    obj_output.sigmoid(),
                    cls_output.sigmoid()
                ], 1)

            outputs.append(output)

        if self.training:

            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )

        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs],
                                dim=2).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs