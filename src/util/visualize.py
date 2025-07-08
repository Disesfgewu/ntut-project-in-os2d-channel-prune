import os2d.utils.visualization as visualizer
from os2d.utils import get_image_size_after_resize_preserving_aspect_ratio

def visualize_boxes_on_image(image_id, boxes_one_image, dataloader, cfg, class_ids=None, path=None, is_detection=False, showfig=True):
    if is_detection and boxes_one_image.has_field("scores"):
        # For detection boxes
        image_to_show = dataloader._get_dataset_image_by_id(image_id)
        h, w = get_image_size_after_resize_preserving_aspect_ratio(h=image_to_show.size[1],
                                                               w=image_to_show.size[0],
                                                               target_size=1500)
        print(f"image to show size: {image_to_show.size}, res")
        image_to_show = image_to_show.resize((w, h))
        visualizer.show_annotated_image(
            img=image_to_show,
            boxes=boxes_one_image,
            labels=boxes_one_image.get_field("labels"),
            scores=boxes_one_image.get_field("scores"),
            class_ids=class_ids,
            score_threshold=cfg.visualization.eval.score_threshold,
            max_dets=cfg.visualization.eval.max_detections,
            showfig=showfig,
            image_id=image_id,
            path=path
        )
    else:
        # For ground truth boxes
        visualizer.show_gt_boxes(
            image_id=image_id,
            gt_boxes=boxes_one_image,
            dataloader=dataloader,
            class_ids=class_ids,
            show_img=showfig,
            path=path
        )