import os
import argparse

import torch

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
from main import parse_opts, init_logger
from os2d.utils import get_image_size_after_resize_preserving_aspect_ratio
from src.util.detection import generate_detection_boxes
from src.util.visualize import visualize_boxes_on_image

# 創建視覺化輸出目錄
visual_dir = Path("./visualized_images")
output_dir = Path("./visualized_images_detection")
visual_dir.mkdir(exist_ok=True, parents=True)
output_dir.mkdir(exist_ok=True, parents=True)

# 預定義顏色映射 (class_id -> BGR顏色)
predefined_color_map = {
    0: (0, 0, 255),       # 紅色: class 0
    1: (0, 255, 0),       # 綠色: class 1
    2: (255, 0, 0),       # 藍色: class 2
    3: (0, 255, 255),     # 黃色: class 3
    4: (255, 0, 255),     # 紫色: class 4
    5: (255, 255, 0),     # 青色: class 5
    6: (0, 165, 255),     # 橙色: class 6
    7: (128, 0, 128),     # 深紫色: class 7
    8: (0, 128, 128),     # 深青色: class 8
    9: (128, 128, 0),     # 橄欖色: class 9
    10: (0, 165, 255),    # 藍橙色: class 10
    11: (255, 99, 71),    # 番茄色: class 11
    13: (50, 205, 50),    # 酸橙綠: class 13
    14: (186, 85, 211),   # 中紫色: class 14
    15: (147, 112, 219),  # 中紫紅色: class 15
    16: (255, 140, 0),    # 深橙色: class 16
    17: (70, 130, 180),   # 鋼藍色: class 17
    18: (210, 105, 30),   # 巧克力色: class 18
    19: (106, 90, 205),   # 石板藍: class 19
    20: (60, 179, 113),   # 海洋綠: class 20
    1055: (0, 255, 127),  # 春綠色: class 1055
    1002: (255, 99, 71)   # 番茄色: class 1002
}

# 擴展顏色池 (20種不同顏色)
color_palette = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),          # 三原色
    (255, 255, 0), (255, 0, 255), (0, 255, 255),    # 二次色
    (128, 0, 128), (128, 128, 0), (0, 128, 128),    # 深色組合
    (255, 165, 0), (255, 192, 203), (173, 216, 230), # 橙/粉/藍
    (0, 128, 0), (128, 0, 0), (0, 0, 128),          # 深綠/紅/藍
    (255, 140, 0), (147, 112, 219), (64, 224, 208), # 深橙/紫/青
    (220, 20, 60), (0, 250, 154), (186, 85, 211)    # 深紅/綠/紫
]


def main():
    cfg, config_file = parse_opts()
    # cfg.init.model = "models/os2d_v2-train.pth"
    init_logger(cfg, config_file)

    # set this to use faster convolutions
    if cfg.is_cuda:
        assert torch.cuda.is_available(), "Do not have available GPU, but cfg.is_cuda == 1"
        torch.backends.cudnn.benchmark = True

    # random seed
    set_random_seed(cfg.random_seed, cfg.is_cuda)

    # Model
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
    
    # start training (validation is inside)
    # trainval_loop(dataloader_train, net, cfg, criterion, optimizer, dataloaders_eval=dataloaders_eval)
    map_of_classes_per_image_id = {}
    with open("text2.txt", "w") as f:
        # 1. 寫入 dataloader 的基本 API 資訊
        f.write("="*80 + "\n")
        f.write("Dataloader API Information\n")
        f.write("="*80 + "\n")
        f.write(f"Total batches: {len(dataloader_train)}\n")
        f.write(f"Batch size: {dataloader_train.batch_size}\n")
        f.write(f"Image normalization: {dataloader_train.img_normalization}\n")
        f.write(f"Class batch size: {dataloader_train.max_batch_labels}\n")
        f.write(f"Data augmentation: {'Enabled' if dataloader_train.data_augmentation else 'Disabled'}\n\n")
        
        # 獲取 dataset 參照 (用於類別名稱映射)
        dataset = dataloader_train.dataset
        
        # 2. 處理前5個批次
        for k in range(5):
            batch = dataloader_train.get_batch(k)
            images, class_images, loc_targets, cls_targets, class_ids, class_sizes, transforms, boxes, img_sizes = batch
            
            # 3. 寫入批次標題
            f.write("\n" + "="*80 + "\n")
            f.write(f"Batch {k} - Image Annotations\n")
            f.write("="*80 + "\n")
            
            # 4. 獲取當前批次的所有圖像ID
            image_ids = dataloader_train.get_image_ids_for_batch_index(k)
            
            # 5. 寫入批次級別資訊
            f.write(f"\nBatch-Level Information:\n")
            f.write(f"Class IDs in batch: {class_ids}\n")
            f.write(f"Image tensor shape: {images.shape}\n")
            f.write(f"Localization targets shape: {loc_targets.shape}\n")
            f.write(f"Classification targets shape: {cls_targets.shape}\n")
            
            # 6. 為每個圖像寫入標註資訊
            for img_id in image_ids:
                map_of_classes_per_image_id[img_id] = {}
                annotation = dataloader_train.get_image_annotation_for_imageid(img_id)
                
                # 寫入圖像ID標題
                f.write(f"\n[Image ID: {img_id}]\n")
                f.write(f"Object count: {len(annotation)}\n")
                
                # 寫入每個物件的詳細資訊
                for obj_idx in range(len(annotation)):
                    # 邊界框座標 (x_min, y_min, x_max, y_max)
                    bbox = annotation.bbox_xyxy[obj_idx].tolist()
                    
                    # 物件類別ID
                    class_id = annotation.get_field('labels')[obj_idx].item()
                    map_of_classes_per_image_id[img_id][class_id] = map_of_classes_per_image_id[img_id].get(class_id, 0) + 1
                    # 獲取類別名稱 (如果可用)
                    class_name = "N/A"
                    if hasattr(dataset, 'get_class_name'):
                        try:
                            class_name = dataset.get_class_name(class_id)
                        except:
                            class_name = f"Unknown (ID: {class_id})"
                    elif hasattr(dataset, 'class_id_to_name'):
                        class_name = dataset.class_id_to_name.get(class_id, f"Unknown (ID: {class_id})")
                    
                    # 困難標記 (如果存在)
                    difficult = ""
                    if annotation.has_field('difficult'):
                        is_difficult = annotation.get_field('difficult')[obj_idx].item()
                        difficult = f", Difficult: {bool(is_difficult)}"
                    
                    # 寫入單行物件資訊
                    f.write(f"  Object {obj_idx+1}: Class={class_id} ({class_name}), "
                            f"BBox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]"
                            f"{difficult}\n")
                    
                    img_path = f"./data/grozi/src/3264/{img_id}.jpg"
                    visualize_boxes_on_image(
                        image_id=img_id,
                        boxes_one_image=annotation,
                        dataloader=dataloader_train,
                        cfg=cfg,
                        class_ids=class_ids
                    )
                for class_id, count in map_of_classes_per_image_id[img_id].items():
                    print(f"Image ID: {img_id}, Class ID: {class_id}, Count: {count}")
                    get , labels , scores = generate_detection_boxes(dataloader_train, net, img_normalization, box_coder, img_id, class_id, cfg, class_num=count)
                    # image_path = f"./visualized_images/{img_id}.jpg"
                    from os2d.modeling.box_coder import BoxList
                    
                    # Get the original image size from dataloader for proper bounding box scaling
                    original_image = dataloader_train._get_dataset_image_by_id(img_id)
                    image_width, image_height = get_image_size_after_resize_preserving_aspect_ratio(h=original_image.size[1],
                                                                                                    w=original_image.size[0],
                                                                                                    target_size=1500)
                    
                    # Create BoxList with proper image dimensions
                    box_list = BoxList(get, (image_width, image_height), mode="xyxy")
                    box_list.add_field("labels", labels)
                    box_list.add_field("scores", scores)  # Add scores field for proper visualization
                    visualize_boxes_on_image(
                        image_id=img_id,
                        boxes_one_image=box_list,
                        dataloader=dataloader_train,
                        cfg=cfg,
                        class_ids=class_ids,
                        path="detection",
                        is_detection=True  # Specify this is detection visualization
                    )
                    # f.write(f"Image ID: {img_id}, Class ID: {class_id}, Get: {get}\n")

            print(f"Processed batch {k} with {len(image_ids)} images")
        # print(f"map_of_classes_per_image_id: {map_of_classes_per_image_id}\n")
        print("Dataloader information saved to text2.txt")

# if __name__ == "__main__":
main()
