# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import unittest

import numpy as np

from sahi.utils.cv import read_image
from sahi.utils.mmdet import MmdetTestConstants, download_mmdet_cascade_mask_rcnn_model, download_mmdet_yolox_tiny_model

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.3
IMAGE_SIZE = 320


def test_perform_inference_with_mask_output():
    from sahi.model import MmdetDetectionModel

    mmdet_detection_model = MmdetDetectionModel(
        model_path=r"/work/Swin-Transformer-Object-Detection/work_dirs/run/train2/epoch_3200.pth",
        config_path=r"/work/Swin-Transformer-Object-Detection/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py",
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device=MODEL_DEVICE,
        category_remapping=None,
        load_at_init=True,
        image_size=IMAGE_SIZE,
    )

    # prepare image
    image_path = "/work/data/train/HG855G_20220822_NG_RJKL71_0000_10_2_1_1.bmp"
    image = read_image(image_path)

    # perform inference
    mmdet_detection_model.perform_inference(image)
    original_predictions = mmdet_detection_model.original_predictions

    boxes = original_predictions[0][0]
    masks = original_predictions[0][1]

    print(len(boxes), len(masks))
    # # ensure all prediction scores are greater then 0.5
    # for box in boxes[0]:
    #     if len(box) == 5:
    #         if box[4] > 0.5:
    #             break



if __name__ == "__main__":
    test_perform_inference_with_mask_output()
