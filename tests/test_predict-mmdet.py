import os
import shutil, json
import unittest
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
import os.path as osp
from pathlib import Path
from sahi.utils.cv import read_image
from sahi.model import MmdetDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction

os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
MODEL_DEVICE = "cuda:0"
# MODEL_DEVICE = "cpu"

CONFIDENCE_THRESHOLD = 0.3
# IMAGE_SIZE = 512


slice_height = 1600
slice_width = 1600
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2
postprocess_type = "GREEDYNMM"

match_metric = "IOS"
match_threshold = 0.1
class_agnostic = True

model_path = r"/work/Swin-Transformer-Object-Detection/work_dirs/run/train-all7-amp/epoch_302.pth"
config_path = r"/work/Swin-Transformer-Object-Detection/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py"


## 初始化模型
mmdet_detection_model = MmdetDetectionModel(
    model_path=model_path,
    config_path=config_path,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    device=MODEL_DEVICE,
    category_remapping=None,
    load_at_init=False,
)
mmdet_detection_model.load_model()


def create_dir(folder, del_existence=False):
    """
        创建指定路径并返回创建的路径

    :param folder: 需创建的路径
    :param del_existence: 是否删除已存在的文件夹
    :return: 输入的路径
    """

    if not isinstance(del_existence, bool):
        raise ValueError('del_existence is bool')

    try:
        if del_existence and os.path.exists(folder):
            shutil.rmtree(folder)
    except Exception:
        pass

    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
    except FileExistsError:
        # TODO: 多进程或多线程调用时，需要优化
        pass

    return folder


def save_json(data, path, name, removed=False):
    """
        将字典数据保存成json文件

    :param data: 字典数据
    :param path: json文件保存文件夹
    :param name: json文件保存名称，当名称重复时会自动增加时间后缀
    :param removed: 是否移除已存在的文件
    :return: 保存的路径
    """

    if data is None or len(data) == 0:
        return

    create_dir(path)
    name = name.replace('.json', '') if name.endswith('.json') else name
    save_path = os.path.join(path, '{}.json'.format(name))
    if os.path.exists(save_path):
        if not removed:
            import time
            cur_time = time.strftime('%m%d%H%M', time.localtime(time.time()))
            save_path = os.path.join(path, '{}_{}.json'.format(name, cur_time))
        else:
            os.remove(save_path)
    print(f'{name}.json is saving...')
    with open(save_path, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    print('save successfully! ->PATH: {}'.format(save_path))
    return save_path


def mask_to_polygons(mask):
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
    mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
    # We add 0.5 to turn them into real-value coordinate space. A better solution
    # would be to first +0.5 and then dilate the returned polygon by 0.5.
    res = [x + 0.5 for x in res if len(x) >= 6]
    return res, has_holes


class TestPredict():

    def test_get_prediction_mmdet(self, image_path):
        # prepare image
        # image = read_image(image_path)
        # get full sized prediction
        prediction_result = get_prediction(
            image=image_path, detection_model=mmdet_detection_model, shift_amount=[0, 0], full_shape=None
        )
        object_prediction_list = prediction_result.object_prediction_list
        return object_prediction_list


    def test_get_sliced_prediction_mmdet(self, image_path):
        # get sliced prediction
        prediction_result = get_sliced_prediction(
            image=image_path,
            detection_model=mmdet_detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            perform_standard_pred=False,
            postprocess_type=None,
            postprocess_match_threshold=match_threshold,
            postprocess_match_metric=match_metric,
            postprocess_class_agnostic=class_agnostic,
        )
        object_prediction_list = prediction_result.object_prediction_list
        return object_prediction_list

 
if __name__ == "__main__":
    test = TestPredict()
    src = r'/work/data/855G3/test/imgs'
    dst = r'/work/data/855G3/inf_test'

    # src = r"/work/data/855G/test/KL-test"
    # dst = r"/work/data/855G/test/inf_test"

    Path(dst).mkdir(parents=True, exist_ok=True)
    imps = glob(src + "/*.bmp")

    inf_jsd = {}
    for imp in tqdm(imps):
        imn = osp.basename(imp)
        print("当前推理图片", imn)
        # 标准推理
        # object_prediction_list = test.test_get_prediction_mmdet(imp) 

        # 切分推理
        object_prediction_list = test.test_get_sliced_prediction_mmdet(imp)
        # print(len(object_prediction_list))
        bboxes, scores, categorys, masks = [], [], [], []
        inf_jsd[imn] = {"filename":imn, "regions":[], "type":"inf"}
        regions = []
        # print("-------dfdf---",len(object_prediction_list))
        for object_prediction in object_prediction_list:
            region = {"shape_attributes":{}, "region_attributes":{}}
            bbox = object_prediction.bbox.to_voc_bbox()
            score = round(object_prediction.score.value, 4)
            category = object_prediction.category.id
            # mask = GenericMask(object_prediction.mask.bool_mask, h, w).polygons
            res, _ = mask_to_polygons(object_prediction.mask.bool_mask)
            mask = [m.reshape(-1, 2) for m in res]
            for m in mask:
                region = {"shape_attributes":{}, "region_attributes":{}}
                xs = list(map(int, m[:,0]))
                ys = list(map(int, m[:,1]))       
                region["shape_attributes"]["all_points_x"] = xs
                region["shape_attributes"]["all_points_y"] = ys
                region["region_attributes"]["regions"] = str(category+1)
                region["region_attributes"]["score"] = str(round(score, 5))
                regions.append(region)
                
        inf_jsd[imn]["regions"] = regions
            
    save_json(inf_jsd, dst, "sahi_inf-crop-1600", removed=True)
