# -*- coding: UTF-8 -*-
# @Author: qixitan
# @Time: 2022/11/4

import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import cv2
import numpy as np
import requests
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image
import torchvision
import argparse
import os
from yolori.exp import get_exp


def make_parser():
    parser = argparse.ArgumentParser("YOLORI cam parser")
    parser.add_argument("-n", "--name", type=str, default="yolox_dior_n", help="model name")
    parser.add_argument("-i", "--imgpath", type=str, default="../imgs/01799.jpg", help="image path")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="device for cam")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    return parser


def main(exp, args):
    model = exp.get_model()
    img_path = args.imgpath
    target_layers = [model.head.obj_preds[2], model.head.reg_preds[2],
                     model.head.cls_preds[2]]  # You can modify it to suit your needs
    device = torch.device(args.device if args.device is not None else 'cpu')

    state_path = args.ckpt
    if state_path is None:
        project_path = "/".join(os.path.abspath(__file__).split("/")[:-2])
        state_path = os.path.join(project_path, "exp_outputs", args.name)
        state_path = os.path.join(state_path, max(os.listdir(state_path)), "best_ckpt.pth")

    state = torch.load(state_path, map_location=device)
    model.eval().to(device)
    model.load_state_dict(state["model"])

    img = np.array(Image.open(img_path))
    # rgb_img = img.copy()
    img = np.float32(img) / 255
    # img = np.float32(img)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform(img).unsqueeze(0)
    cam = EigenCAM(model, target_layers, use_cuda=False)
    grayscale_cam = cam(tensor)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    x2 = Image.fromarray(cam_image)
    x2.save(img_path.replace(".jpg", "_EigenCam_{}.jpg").format(args.name))


if __name__ == '__main__':
    args = make_parser().parse_args()
    exp = get_exp(exp_name=args.name)
    main(exp=exp, args=args)

