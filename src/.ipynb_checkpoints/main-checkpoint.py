import os
import argparse

import torch

from os2d.modeling.model import build_os2d_from_config

from os2d.data.dataloader import build_eval_dataloaders_from_cfg, build_train_dataloader_from_config
from os2d.engine.train import trainval_loop
from os2d.utils import set_random_seed, get_trainable_parameters, mkdir, save_config, setup_logger, get_data_path
from os2d.engine.optimization import create_optimizer
from os2d.config import cfg

import os
import argparse

import torch
import os
import argparse
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

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

import torch

def show_gpu_memory_usage():
    """顯示當前 GPU RAM 使用情況"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)

        # 獲取記憶體使用情況 (轉換為 GB)
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3

        print(f"🖥️  GPU 設備: {device_name}")
        print(f"📊 記憶體使用情況:")
        print(f"   已分配: {allocated:.2f} GB")
        print(f"   已保留: {reserved:.2f} GB")
        print(f"   總容量: {total:.2f} GB")
        print(f"   使用率: {(allocated/total)*100:.1f}%")
        print(f"   保留率: {(reserved/total)*100:.1f}%")

        # 視覺化進度條
        usage_percent = int((allocated/total)*100)
        bar_length = 20
        filled_length = int(bar_length * usage_percent / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"   [{bar}] {usage_percent}%")

    else:
        print("❌ CUDA 不可用，無法檢測 GPU 記憶體")

import torch

def count_parameters(model):
    """
    計算模型的參數數量
    Returns:
      total_params: 包含所有參數
      trainable_params: 只包含 requires_grad=True 的參數
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

import io

def estimate_model_size(model):
    """
    將模型序列化到緩衝區，估算存檔大小（MB）
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.getbuffer().nbytes / (1024 ** 2)
    return size_mb

def show_network_status( orig_net, prune_net ):
    print( "原始網路參數統計:\n" )
    model = orig_net
    total, trainable = count_parameters(model)
    print(f"總參數量: {total:,}")
    print(f"可訓練參數量: {trainable:,}")

    size_mb = estimate_model_size(model)
    print(f"模型存儲大小: {size_mb:.2f} MB")

    print( "剪枝網路參數統計:\n" )
    model = prune_net
    total, trainable = count_parameters(model)
    print(f"總參數量: {total:,}")
    print(f"可訓練參數量: {trainable:,}")

    size_mb = estimate_model_size(model)
    print(f"模型存儲大小: {size_mb:.2f} MB")


def parse_opts():
    parser = argparse.ArgumentParser(description="Training and evaluation of the OS2D model")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_known_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return cfg, args.config_file


def init_logger(cfg, config_file):
    output_dir = cfg.output.path
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("OS2D", output_dir if cfg.output.save_log_to_file else None)

    if config_file:
        logger.info("Loaded configuration file {}".format(config_file))
        with open(config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    else:
        logger.info("Config file was not provided")

    logger.info("Running with config:\n{}".format(cfg))

    # save config file only when training (to run multiple evaluations in the same folder)
    if output_dir and cfg.train.do_training:
        output_config_path = os.path.join(output_dir, "config.yml")
        logger.info("Saving config into: {}".format(output_config_path))
        # save overloaded model config in the output directory
        save_config(cfg, output_config_path)




def main():
    cfg, config_file = parse_opts()
    init_logger(cfg, config_file)

    # set this to use faster convolutions
    if cfg.is_cuda:
        assert torch.cuda.is_available(), "Do not have available GPU, but cfg.is_cuda == 1"
        torch.backends.cudnn.benchmark = True

    # random seed
    set_random_seed(cfg.random_seed, cfg.is_cuda)

    # Model
    cfg.defrost()
    cfg.init.model = './src/util/checkpoints-test/checkpoint_lcp_finetune_26_layer3.5.conv2.pth'
    cfg.freeze()
    net, box_coder, criterion, img_normalization, optimizer_state = build_os2d_from_config(cfg)

    # Optimizer
    parameters = get_trainable_parameters(net)
    optimizer = create_optimizer(parameters, cfg.train.optim, optimizer_state)

    # load the dataset
    data_path = get_data_path()
    dataloader_train, datasets_train_for_eval = build_train_dataloader_from_config(cfg, box_coder, img_normalization,
                                                                                   data_path=data_path)

    dataloaders_eval = build_eval_dataloaders_from_cfg(cfg, box_coder, img_normalization,
                                                       datasets_for_eval=datasets_train_for_eval,
                                                       data_path=data_path)

    db = DataLoaderDB( path = './src/db/data.csv' , dataloader = dataloader_train)


    from src.lcp.ct_aoi_align import ContextAoiAlign
    transform_image = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(img_normalization["mean"], img_normalization["std"])
                        ])

    context_aoi_align = ContextAoiAlign( db, dataloader_train, transform_image , net , cfg )

    from src.lcp.aux_net import AuxiliaryNetwork
    aux_net = AuxiliaryNetwork( context_aoi_align, db )

    from src.util.prune_db import PruneDBControler
    prune_db = PruneDBControler( path = './src/db/prune_channel_information1.csv' )

    from src.lcp.lcp import LCP
    lcp = LCP(net, aux_net, dataloader_train)
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

    for name, ch in layers:
        if name == 'layer1.0.conv1':
            pass
        else:
            continue
        print(f"{name}: {ch} channels")
        keep, discard = lcp.get_channel_selection_by_no_grad(
            layer_name   = f"net_feature_maps.{name}",
            discard_rate = 0.5,
            lambda_rate  = 1.0,
            use_image_num= 3,
            random_seed  = 42
        )
        print(f"layer {name} , 預計保留通道數量: {len(keep)}/{ch}, 預計捨棄通道數量: {len(discard)}/{ch}")

    for layer in pruned_layers:
        lcp.prune_layer(
            layer_name   = layer,
            discard_rate = None,
        )

    show_network_status( net, lcp._prune_net )

    net = lcp._prune_net

    # start training (validation is inside)
    trainval_loop(dataloader_train, net, cfg, criterion, optimizer, dataloaders_eval=dataloaders_eval)

main()