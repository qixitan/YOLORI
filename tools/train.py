#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolori.core import Trainer, launch
from yolori.exp import get_exp
from yolori.utils import configure_nccl, configure_omp, get_num_devices

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5, 7"


def make_parser():
    parser = argparse.ArgumentParser("YOLORI train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str,  help="model name")
    parser.add_argument("-b", "--batch-size", type=int, help="batch size")
    parser.add_argument("-d", "--devices", type=int, help="device for training")

    parser.add_argument("-r", "--resume", action="store_true", help="resume training")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("-e", "--start_epoch", default=None, type=int, help="resume training start epoch", )

    parser.add_argument("-f", "--exp_file", default=None, type=str, help="plz input your experiment description file", )
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training", )
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")

    parser.add_argument("-o", "--occupy", dest="occupy", action="store_true",
                        help="occupy GPU memory first for training.", )
    parser.add_argument("--cache", dest="cache", action="store_true",
                        help="Caching imgs to RAM for fast training.", )
    parser.add_argument("--fp16", dest="fp16", action="store_true",
                        help="Adopting mix precision training.", )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER, )
    return parser



@logger.catch
def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    model = exp.get_model()
    # print(model)
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )
