import torchvision.transforms as transforms
import os2d.structures.transforms as transforms_boxes
import os2d.utils.visualization as visualizer
import torch
from os2d.utils import get_image_size_after_resize_preserving_aspect_ratio
from os2d.structures.feature_map import FeatureMapSize

def generate_detection_boxes(dataloader, net, img_normalization , box_coder, image_id, class_id, cfg, class_num=0):
    """
    生成指定圖像和類別的檢測框
    
    Args:
        dataloader: 數據加載器 (DataloaderOneShotDetection 實例)
        net: 訓練好的模型
        image_id: 目標圖像ID
        class_id: 目標類別ID
        image_size: 可選的圖像處理尺寸
        class_num: 類別數量 (用於過濾)
    
    Returns:
        BoxList: 包含檢測框的對象
    """
    # 1. 獲取圖像數據
    input_image = dataloader._get_dataset_image_by_id(image_id)
    class_img = dataloader.dataset.gt_images_per_classid[class_id]
    def ensure_rgb(image):
        if image.mode == 'L':  # 灰度图转RGB
            return image.convert('RGB')
        elif image.mode == 'RGBA':  # 带透明度图转RGB
            return image.convert('RGB')
        return image
    
    input_image = ensure_rgb(input_image)
    class_img = ensure_rgb(class_img)

    transform_image = transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize(img_normalization["mean"], img_normalization["std"])
                      ])

    h, w = get_image_size_after_resize_preserving_aspect_ratio(h=input_image.size[1],
                                                               w=input_image.size[0],
                                                               target_size=1500)
    # print(f"Original image size: {input_image.size}, resized to: {(w, h)}")
    input_image = input_image.resize((w, h))

    input_image_th = transform_image(input_image)
    input_image_th = input_image_th.unsqueeze(0)
    if cfg.is_cuda:
        input_image_th = input_image_th.cuda()
    h, w = get_image_size_after_resize_preserving_aspect_ratio(h=class_img.size[1],
                                                               w=class_img.size[0],
                                                               target_size=cfg.model.class_image_size)
    class_img = class_img.resize((w, h))
    class_image_th = transform_image(class_img)
    if cfg.is_cuda:
        class_image_th = class_image_th.cuda()

    # 6. 前向傳播
    # print( f"Input image shape: {input_image_th.shape}, class image shape: {class_image_th.shape}")
    with torch.no_grad():
        loc_prediction_batch, class_prediction_batch, _, fm_size, transform_corners_batch = net(images=input_image_th, class_images=[class_image_th])
    
    image_loc_scores_pyramid = [loc_prediction_batch[0]]
    image_class_scores_pyramid = [class_prediction_batch[0]]
    img_size_pyramid = [FeatureMapSize(img=input_image_th)]
    transform_corners_pyramid = [transform_corners_batch[0]]

    boxes = box_coder.decode_pyramid(image_loc_scores_pyramid, image_class_scores_pyramid,
                                           img_size_pyramid, [class_id],
                                           nms_iou_threshold=cfg.eval.nms_iou_threshold,
                                           nms_score_threshold=cfg.eval.nms_score_threshold,
                                           transform_corners_pyramid=transform_corners_pyramid)

    # remove some fields to lighten visualization                                       
    boxes.remove_field("default_boxes")
    # print( f"Boxes image size: {boxes.image_size}")
    # Note that the system outputs the correaltions that lie in the [-1, 1] segment as the detection scores (the higher the better the detection).
    scores = boxes.get_field("scores")
    
    # print( boxes )
    # print( boxes.get_field("labels") )
    # print( scores )

    cfg.defrost()
    cfg.visualization.eval.max_detections = class_num
    cfg.visualization.eval.score_threshold = float("-inf")
    cfg.freeze()
    boxes, labels, scores = visualizer.show_detections(boxes, input_image,
                            cfg.visualization.eval, show_fig=False)
    # print(f"Image ID: {image_id}, Class ID: {class_id}, Boxes: {boxes}, Scores: {scores}, Labels: {labels}\n")
    
    
    # Scale boxes to match original image dimensions
    scaled_boxes = boxes.clone()
    return scaled_boxes, labels, scores