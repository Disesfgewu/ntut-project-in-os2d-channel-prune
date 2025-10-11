import os
import argparse

import torch
import os
import argparse
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

from os2d.modeling import box_coder
from os2d.modeling.model import build_os2d_from_config
from os2d.config import cfg
import  os2d.utils.visualization as visualizer
from os2d.structures.feature_map import FeatureMapSize
from os2d.utils import setup_logger, read_image, get_image_size_after_resize_preserving_aspect_ratio
from os2d.data import dataloader
from os2d.modeling.model import build_os2d_from_config

from os2d.data.dataloader import build_eval_dataloaders_from_cfg, build_train_dataloader_from_config
from os2d.engine.train import trainval_loop
from os2d.utils import set_random_seed, get_trainable_parameters, mkdir, save_config, setup_logger, get_data_path
from os2d.engine.optimization import create_optimizer
from os2d.config import cfg
from os2d.utils.visualization import *
import random
import os2d.utils.visualization as visualizer
from pathlib import Path
import cv2
import numpy as np
from os2d.utils import get_image_size_after_resize_preserving_aspect_ratio
from src.util.detection import generate_detection_boxes
from src.util.visualize import visualize_boxes_on_image
from src.util.filter import DataLoaderDB
logger = setup_logger("OS2D")

def init_prune_model( model_path , channel_information_path="./src/db/prune_channel_information.csv"):
    # use GPU if have available
    cfg.defrost()
    
    cfg.is_cuda = torch.cuda.is_available()

    if cfg.is_cuda:
        assert torch.cuda.is_available(), "Do not have available GPU, but cfg.is_cuda == 1"
        torch.backends.cudnn.benchmark = True

    # random seed
    set_random_seed(cfg.random_seed, cfg.is_cuda)

    # Model
    cfg.init.model = model_path
    cfg.freeze()

    # net = torch.load( cfg.init.model, map_location=torch.device('cuda') )
    net, box_coder, criterion, img_normalization, optimizer_state = build_os2d_from_config(cfg, channel_information_path=channel_information_path)
    if not model_path.endswith("os2d_v2-train.pth"):
        from src.util.prune_db import PruneDBControler
        prune_db = PruneDBControler( path = channel_information_path )

        from src.lcp.lcp import LCP
        lcp = LCP(net)
        lcp.init_for_indices()
        lcp.set_prune_db( prune_db )

        from src.lcp.pruner import Pruner
        pruner = Pruner( lcp._prune_net )
        pruner.set_prune_db( prune_db )

        layers = prune_db.get_all_layers()
        pruned_layers = []
        for layer in layers:
            if layer not in pruned_layers and layer.startswith('layer'):
                pruned_layers.append(layer)

                layers = lcp.get_layers_name()

        for layer in pruned_layers:
            lcp.prune_layer(
                layer_name   = layer,
                discard_rate = None,
            )

        net = lcp._prune_net.cuda()

    return net, img_normalization, box_coder

def detection(net, box_coder, img_normalization, t_input_image, t_class_images):
    # 前處理
    input_image_pil = read_image(t_input_image)
    w, h = get_image_size_after_resize_preserving_aspect_ratio(
        h=input_image_pil.size[1],
        w=input_image_pil.size[0],
        target_size=1500
    )
    input_image_pil = input_image_pil.resize((w, h))

    transform_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(img_normalization["mean"], img_normalization["std"])
    ])

    device = torch.device('cuda') if cfg.is_cuda else torch.device('cpu')
    input_image_th = transform_image(input_image_pil).unsqueeze(0).to(device)

    # 避免重複 read_image
    class_images_pil = []
    for class_img_path in t_class_images:
        img = read_image(class_img_path)
        cw, ch = get_image_size_after_resize_preserving_aspect_ratio(
            h=img.size[1], w=img.size[0], target_size=cfg.model.class_image_size
        )
        class_images_pil.append(img.resize((cw, ch)))
    class_images_th = [transform_image(img).to(device) for img in class_images_pil]
    class_ids = list(range(len(t_class_images)))

    # 推論
    with torch.no_grad():
        loc_pred, class_pred, _, fm_size, transform_corners = net(images=input_image_th, class_images=class_images_th)
    image_loc_scores_pyramid = [loc_pred[0]]
    image_class_scores_pyramid = [class_pred[0]]
    img_size_pyramid = [FeatureMapSize(img=input_image_th)]
    transform_corners_pyramid = [transform_corners[0]]

    boxes = box_coder.decode_pyramid(
        image_loc_scores_pyramid, image_class_scores_pyramid, img_size_pyramid, class_ids,
        nms_iou_threshold=cfg.eval.nms_iou_threshold,
        nms_score_threshold=cfg.eval.nms_score_threshold,
        transform_corners_pyramid=transform_corners_pyramid,
    )
    boxes.remove_field("default_boxes")
    # 不做 plt.figure / subplot / imshow
    cfg.defrost()
    cfg.visualization.eval.max_detections = len(t_class_images)
    cfg.visualization.eval.score_threshold = 0.10
    cfg.freeze()

    # 直接回傳視覺化 numpy 或 PIL
    result_img, info = visualizer.show_detections(
        boxes, input_image_pil, cfg.visualization.eval, ifdetection=True
    )
    return result_img, info