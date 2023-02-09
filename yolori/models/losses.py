#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import math

import torch.nn.functional as F
from torch.autograd import Variable


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        """
        :param pred: x_c y_c w h
        :param target: x_c y_c w h
        :return:
        """
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0) ** 2
        elif self.loss_type == 'diou':
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )

            w_c = (c_br - c_tl)[:, 0]
            h_c = (c_br - c_tl)[:, 1]
            c = w_c ** 2 + h_c ** 2

            w_d = (pred[:, :2] - target[:, :2])[:, 0]
            h_d = (pred[:, :2] - target[:, :2])[:, 1]
            d = w_d ** 2 + h_d ** 2

            diou = iou - d / c
            loss = 1 - diou.clamp(min=-1.0, max=1.0) ** 2
        elif self.loss_type == "ciou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            convex_dis = torch.pow(c_br[:, 0] - c_tl[:, 0], 2) + torch.pow(c_br[:, 1] - c_tl[:, 1],
                                                                           2) + 1e-7  # convex diagonal squared

            center_dis = (torch.pow(pred[:, 0] - target[:, 0], 2) + torch.pow(pred[:, 1] - target[:, 1],
                                                                              2))  # center diagonal squared

            v = (4 / math.pi ** 2) * torch.pow(torch.atan(target[:, 2] / torch.clamp(target[:, 3], min=1e-7)) -
                                               torch.atan(pred[:, 2] / torch.clamp(pred[:, 3], min=1e-7)), 2)

            with torch.no_grad():
                alpha = v / ((1 + 1e-7) - iou + v)

            ciou = iou - (center_dis / convex_dis + alpha * v)

            loss = 1 - ciou.clamp(min=-1.0, max=1.0) ** 2
        elif self.loss_type == "eiou":

            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            convex_dis = torch.pow(c_br[:, 0] - c_tl[:, 0], 2) + torch.pow(c_br[:, 1] - c_tl[:, 1],
                                                                           2) + 1e-7  # convex diagonal squared

            center_dis = (torch.pow(pred[:, 0] - target[:, 0], 2) + torch.pow(pred[:, 1] - target[:, 1],
                                                                              2))  # center diagonal squared

            dis_w = torch.pow(pred[:, 2] - target[:, 2], 2)  # 两个框的w欧式距离
            dis_h = torch.pow(pred[:, 3] - target[:, 3], 2)  # 两个框的h欧式距离

            C_w = torch.pow(c_br[:, 0] - c_tl[:, 0], 2) + 1e-7  # 包围框的w平方
            C_h = torch.pow(c_br[:, 1] - c_tl[:, 1], 2) + 1e-7  # 包围框的h平方

            eiou = iou - (center_dis / convex_dis) - (dis_w / C_w) - (dis_h / C_h)

            loss = 1 - eiou.clamp(min=-1.0, max=1.0) ** 2

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=False):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class VarifocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 alpha=0.75,
                 gamma=2.0,
                 iou_weighted=True,
                 reduction='mean',
                 loss_weight=1.0):
        """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_
        Args:
            use_sigmoid (bool, optional): Whether the prediction is
                used for sigmoid or softmax. Defaults to True.
            alpha (float, optional): A balance factor for the negative part of
                Varifocal Loss, which is different from the alpha of Focal
                Loss. Defaults to 0.75.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            iou_weighted (bool, optional): Whether to weight the loss of the
                positive examples with the iou target. Defaults to True.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(VarifocalLoss, self).__init__()
        assert use_sigmoid is True, \
            'Only sigmoid varifocal loss supported now.'
        assert alpha >= 0.0
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * self.varifocal_loss(
                pred,
                target,
                weight,
                alpha=self.alpha,
                gamma=self.gamma,
                iou_weighted=self.iou_weighted,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls

    def varifocal_loss(self, pred,
                       target,
                       weight=None,
                       alpha=0.75,
                       gamma=2.0,
                       iou_weighted=True,
                       reduction='mean',
                       avg_factor=None):
        """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_
        Args:
            pred (torch.Tensor): The prediction with shape (N, C), C is the
                number of classes
            target (torch.Tensor): The learning target of the iou-aware
                classification score with shape (N, C), C is the number of classes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            alpha (float, optional): A balance factor for the negative part of
                Varifocal Loss, which is different from the alpha of Focal Loss.
                Defaults to 0.75.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            iou_weighted (bool, optional): Whether to weight the loss of the
                positive example with the iou target. Defaults to True.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
        """
        # pred and target should be of the same size
        assert pred.size() == target.size()
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        if iou_weighted:
            focal_weight = target * (target > 0.0).float() + \
                           alpha * (pred_sigmoid - target).abs().pow(gamma) * \
                           (target <= 0.0).float()
        else:
            focal_weight = (target > 0.0).float() + \
                           alpha * (pred_sigmoid - target).abs().pow(gamma) * \
                           (target <= 0.0).float()
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none') * focal_weight
        loss = self.weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    def weight_reduce_loss(self, loss, weight=None, reduction='mean', avg_factor=None):
        """Apply element-wise weight and reduce loss.
        Args:
            loss (Tensor): Element-wise loss.
            weight (Tensor): Element-wise weights.
            reduction (str): Same as built-in losses of PyTorch.
            avg_factor (float): Avarage factor when computing the mean of losses.
        Returns:
            Tensor: Processed loss values.
        """
        # if weight is specified, apply element-wise weight
        if weight is not None:
            loss = loss * weight

        # if avg_factor is not specified, just reduce the loss
        if avg_factor is None:
            loss = self.reduce_loss(loss, reduction)
        else:
            # if reduction is mean, then average the loss by avg_factor
            if reduction == 'mean':
                loss = loss.sum() / avg_factor
            # if reduction is 'none', then do nothing, otherwise raise an error
            elif reduction != 'none':
                raise ValueError('avg_factor can not be used with reduction="sum"')
        return loss

    def reduce_loss(self, loss, reduction):
        """Reduce loss as specified.
        Args:
            loss (Tensor): Elementwise loss tensor.
            reduction (str): Options are "none", "mean" and "sum".
        Return:
            Tensor: Reduced loss tensor.
        """
        reduction_enum = F._Reduction.get_enum(reduction)
        # none: 0, elementwise_mean:1, sum: 2
        if reduction_enum == 0:
            return loss
        elif reduction_enum == 1:
            return loss.mean()
        elif reduction_enum == 2:
            return loss.sum()


class alpha_IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="ciou", alpha=3):
        super(alpha_IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

        self.alpha = alpha

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-7)

        if self.loss_type == "iou":
            loss = 1 - iou ** self.alpha  ###############   2>>>>3(a-iou)
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou ** self.alpha - ((area_c - area_u) / area_c.clamp(1e-16)) ** self.alpha
            loss = 1 - giou.clamp(min=-1.0, max=1.0)


        elif self.loss_type == "ciou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            convex_dis = torch.pow(c_br[:, 0] - c_tl[:, 0], 2) + torch.pow(c_br[:, 1] - c_tl[:, 1],
                                                                           2) + 1e-7  # convex diagonal squared

            center_dis = (torch.pow(pred[:, 0] - target[:, 0], 2) + torch.pow(pred[:, 1] - target[:, 1],
                                                                              2))  # center diagonal squared

            v = (4 / math.pi ** 2) * torch.pow(torch.atan(target[:, 2] / torch.clamp(target[:, 3], min=1e-7)) -
                                               torch.atan(pred[:, 2] / torch.clamp(pred[:, 3], min=1e-7)), 2)
            with torch.no_grad():
                beat = v / (v - iou + 1)

            ciou = iou ** self.alpha - (center_dis ** self.alpha / convex_dis ** self.alpha + (beat * v) ** self.alpha)

            loss = 1 - ciou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
