from typing import Callable, Union
import tensorflow as tf
import numpy as np
import numpy.typing as npt
from matplotlib import colors
from matplotlib import cm as cmx
import matplotlib.pyplot as plt

from cs_sem_seg.configs import IMAGE_MEAN, IMAGE_STD
from cs_sem_seg.data.cs_data import Cityscapes
from cs_sem_seg.loss import get_pixel_loss


# ------------------------ Color Config ------------------------
jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)


def unnormalize_image(image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return image * IMAGE_STD + IMAGE_MEAN


def get_masked_img(image: npt.NDArray[np.float32], mask: npt.NDArray[np.uint8]) -> np.ndarray:
    excluded_mask = mask.sum(-1) == 0
    if len(mask.shape) > 2:
        if mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)
        else:
            mask = np.argmax(mask, axis=-1)
    mask[excluded_mask] = 19
    return mask


def get_cityscape_mask_img(mask: npt.NDArray[np.uint8]) -> np.ndarray:
    if len(mask.shape) > 2:
        if mask.shape[-1] == 1:
            cat_mask = np.squeeze(mask, axis=-1)
        else:
            cat_mask = np.argmax(mask, axis=-1)  # this introduce 0 at places where no GT is present (zero all channels)
    else:
        cat_mask = mask
    cat_mask[mask.sum(-1) == 0] = 19  # this marks the place with all zeros using idx 19
    mask_image = Cityscapes.decode_target(cat_mask)
    return mask_image


def overlay_loss_img(ls_image, image, on_top=True):
    ls_image = ls_image.clip(0, np.percentile(ls_image, 95))
    ls_image /= ls_image.max()
    heatmap = scalarMap.to_rgba(ls_image)[..., :-1]
    if on_top:
        overlayed_image = ((heatmap * 0.4 + image * 0.6).clip(0,1)*255).astype(np.uint8)
    else:
        overlayed_image = ((heatmap).clip(0, 1) * 255).astype(np.uint8)
    return overlayed_image


def get_cce_loss_overlayed_img(image: npt.NDArray[np.float32], prediction: npt.NDArray[np.float32], gt: npt.NDArray[np.float32]) -> np.ndarray:
    image = unnormalize_image(image)
    ls = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
    ls_image = ls(gt, prediction).numpy()
    return overlay_loss_img(ls_image, image, False)


def get_bce_loss_overlayed_img(image: npt.NDArray[np.float32], prediction: npt.NDArray[np.float32],
                               gt: npt.NDArray[np.float32]
                               ) -> np.ndarray:
    image = unnormalize_image(image)
    ls = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
    ls_image = ls(gt, prediction).numpy()
    return overlay_loss_img(ls_image, image, False)


def get_custom_ce_loss_overlayed_img(image: npt.NDArray[np.float32], prediction: npt.NDArray[np.float32],
                               gt: npt.NDArray[np.float32]
                               ) -> np.ndarray:
    image = unnormalize_image(image)
    ls_image = get_pixel_loss(gt, prediction)
    return overlay_loss_img(ls_image.numpy(), image, False)
