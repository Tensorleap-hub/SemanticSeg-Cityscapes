import tensorflow as tf
import numpy.typing as npt
import json
from typing import Dict, Any, Union
from PIL import Image

from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.visualizer_classes import LeapImage, LeapImageMask
from code_loader.contract.enums import (
    LeapDataType
)
from numpy import ndarray

from cs_sem_seg.configs import *
from cs_sem_seg.data.cs_data import CATEGORIES
from cs_sem_seg.utils.tl_utils import subset_images
from cs_sem_seg.visualizers.visualizers import get_loss_overlayed_img, get_cityscape_mask_img, get_masked_img
from cs_sem_seg.visualizers.visualizers_utils import unnormalize_image
from cs_sem_seg.metrics import mean_iou, class_mean_iou
from cs_sem_seg.utils.kili_utils import _download, get_masks
from cs_sem_seg.configs import IMAGE_SIZE, LOCAL_DIR
from cs_sem_seg.data.cs_data import Cityscapes


# ----------------------------------- Input ------------------------------------------

def non_normalized_input_image(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data[idx]
    kili_external_id = data['externalId']
    # img_url = data['content']
    # fpath = download(kili_external_id, img_url, LOCAL_DIR)
    fpath = _download(kili_external_id)
    img = np.array(Image.open(fpath).convert('RGB').resize(IMAGE_SIZE)) / 255.
    return img


def input_image(idx: int, data: PreprocessResponse) -> np.ndarray:
    img = non_normalized_input_image(idx, data)
    normalized_image = (img - IMAGE_MEAN) / IMAGE_STD
    return normalized_image.astype(float)

# ----------------------------------- GT ------------------------------------------

def ground_truth_mask(idx: int, data: PreprocessResponse) -> np.ndarray:
    kili_external_id = data.data[idx]['externalId']
    mask, _ = get_masks(kili_external_id)
    return mask

# ----------------------------------- Metadata ------------------------------------------

def get_categorical_mask(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    cloud_path = data['gt_path'][idx % data["real_size"]]
    fpath = _download(cloud_path)
    mask = np.array(Image.open(fpath).resize(IMAGE_SIZE, Image.Resampling.NEAREST))
    encoded_mask = Cityscapes.encode_target_cityscapes(mask)
    return encoded_mask


def metadata_idx(idx: int, data: PreprocessResponse) -> int:
    """ add TL index """
    return idx



def metadata_json_data(idx: int, data: PreprocessResponse) -> Dict[str, Union[str, Any]]:
    data = data.data[idx]
    json_data = dict()
    json_data['kili_external_id'] = data['externalId']
    json_data.update(data['jsonMetadata'])
    return json_data



def metadata_class(idx: int, data: PreprocessResponse) -> dict:
    kili_external_id = data.data[idx]['externalId']
    mask, cat_cnt = get_masks(kili_external_id)
    res = dict(zip([f'{c}_obj_cnt' for c in CATEGORIES], cat_cnt))
    for i, c in enumerate(CATEGORIES):
        mask_i = mask[..., i]
        res[f'{c}_percent'] = (mask_i.sum() / mask_i.size).round(3).astype(np.float32)
    return res


def metadata_brightness(idx: int, data: PreprocessResponse) -> ndarray:
    img = non_normalized_input_image(idx, data)
    return np.mean(img)


def metadata_filename_city_dataset(idx: int, data: PreprocessResponse) -> Dict[str, Any]:
    res = {'file_names': data.data['file_names'][idx],
           'cities': data.data['cities'][idx]
           }
    return res


# ----------------------------------- Visualizers ------------------------------------------


def image_visualizer(image: npt.NDArray[np.float32]) -> LeapImage:
    return LeapImage((unnormalize_image(image) * 255).astype(np.uint8))


def mask_visualizer(image: npt.NDArray[np.float32], mask: npt.NDArray[np.uint8]) -> LeapImageMask:
    mask = get_masked_img(image, mask)
    return LeapImageMask(mask.astype(np.uint8), unnormalize_image(image).astype(np.float32), CATEGORIES + ["excluded"])


def cityscape_segmentation_visualizer(mask: npt.NDArray[np.uint8]) -> LeapImage:
    mask_image = get_cityscape_mask_img(mask)
    return LeapImage(mask_image.astype(np.uint8))


def loss_visualizer(image: npt.NDArray[np.float32], prediction: npt.NDArray[np.float32],
                    gt: npt.NDArray[np.float32]) -> LeapImage:
    overlayed_image = get_loss_overlayed_img(image, prediction, gt)
    return LeapImage(overlayed_image)


# ----------------------------------- Binding ------------------------------------------

leap_binder.set_preprocess(subset_images)


leap_binder.set_input(input_image, 'normalized_image')

leap_binder.set_ground_truth(ground_truth_mask, 'mask')

leap_binder.add_custom_metric(class_mean_iou, name=f"iou_class")
leap_binder.add_custom_metric(mean_iou, name=f"iou")

leap_binder.set_metadata(metadata_idx, 'idx')
leap_binder.set_metadata(metadata_json_data, 'json_data')
leap_binder.set_metadata(metadata_class, 'class')
leap_binder.set_metadata(metadata_brightness, 'brightness')

leap_binder.set_visualizer(image_visualizer, 'image_visualizer', LeapDataType.Image)
leap_binder.set_visualizer(mask_visualizer, 'mask_visualizer', LeapDataType.ImageMask)
leap_binder.set_visualizer(cityscape_segmentation_visualizer, 'cityscapes_visualizer', LeapDataType.Image)
leap_binder.set_visualizer(loss_visualizer, 'loss_visualizer', LeapDataType.Image)

leap_binder.add_prediction('seg_mask', CATEGORIES)
