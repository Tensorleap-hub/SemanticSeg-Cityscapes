import os
from typing import Optional, List, Tuple
import urllib.request
from functools import lru_cache
from kili.client import Kili
from kili.services.label_data_parsing.annotation import BoundingPolyAnnotation
from kili.utils.labels.image import normalized_vertices_to_mask
import numpy as np

from cs_sem_seg.configs import KILI_PROJECT_ID, IMAGE_SIZE, NUM_CLASSES, LOCAL_DIR
from cs_sem_seg.data.cs_data import CATEGORIES_IDS


@lru_cache()
def _connect_to_kili() -> Kili:
    return Kili(api_key=os.environ['AUTH_SECRET'])


def _download(kili_external_id: str, kili: Kili = None, local_file_dir: Optional[str] = None, use_cache: bool = True) -> str:

    filename = kili_external_id + ".png"

    # if local_file_path is not specified saving in home dir
    if local_file_dir is None:
        local_file_dir = os.path.join(LOCAL_DIR, KILI_PROJECT_ID, "assets", filename)

    local_file_path = os.path.join(local_file_dir, filename)

    # check if file already exists
    if os.path.exists(local_file_path) and use_cache:
        return local_file_path

    kili = _connect_to_kili() if kili is None else kili

    kili.assets(
        project_id=KILI_PROJECT_ID,
        external_id_strictly_in=kili_external_id,
        download_media=True,
        local_media_dir=local_file_dir
        # fields=[]
        )

    return local_file_path



def _convert_annotation_to_np(annotation: BoundingPolyAnnotation):
    vertices = annotation.bounding_poly[0].normalized_vertices
    return np.array([[vertex['x'], vertex['y']] for vertex in vertices])


def _convert_annotation_to_mask(annotation: BoundingPolyAnnotation):
    # normalized_vertices = annotation.bounding_poly[0].normalized_vertices
    normalized_vertices = annotation['boundingPoly'][0]['normalizedVertices']
    mask = normalized_vertices_to_mask(normalized_vertices, *IMAGE_SIZE).astype(np.int32)
    mask[mask == 255] = 1
    return mask


def get_killi_assets(sub_names: List[str] = ['train', 'val', 'test'], sub_sizes: List[int] = [5, 5, 5]) -> Tuple[List[dict], Kili]:
    assert len(sub_names) == len(sub_sizes)
    kili = _connect_to_kili()
    assets, labels = [], []
    for sub, sub_size in zip(sub_names, sub_sizes):
        assets += [kili.assets(project_id=KILI_PROJECT_ID, skip=0, first=sub_size, metadata_where={'split': sub})]#, download_media=True, local_media_dir='nfs')
        # labels += [kili.labels(project_id=KILI_PROJECT_ID, fields=['jsonResponse'], skip=0, metadata_where={'split': sub})]
    return assets, kili



# def get_masks(kili_external_id: str, kili: Kili = None):
#
#     kili = _connect_to_kili() if kili is None else kili
#
#     labels = kili.labels(
#         project_id=KILI_PROJECT_ID,
#         asset_external_id_in=[kili_external_id],
#         fields=['jsonResponse'],
#         # download_media=True,
#         # local_media_dir='nfs'
#         # output_format='parsed_label',
#     )
#     # annotations = labels[0].jobs["OBJECT_DETECTION_JOB"].annotations
#     annotations = labels[0]['jsonResponse']['OBJECT_DETECTION_JOB']['annotations']
#     cat_cnt = np.zeros(NUM_CLASSES).astype(np.int32)
#     res_mask = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], NUM_CLASSES)).astype(np.int32)
#     for ann in annotations:
#         # cat = ann.category.name.lower().replace('_', ' ')
#         cat = ann['categories'][0]['name'].lower().replace('_', ' ')
#         if cat not in CATEGORIES_IDS:
#             continue
#         cat_i = CATEGORIES_IDS[cat]
#         res_mask[..., cat_i] |= _convert_annotation_to_mask(ann)
#         cat_cnt[cat_i] += 1
#
#     return res_mask, cat_cnt


def get_masks(kili_labels):
    annotations = kili_labels[0]['jsonResponse']['OBJECT_DETECTION_JOB']['annotations']
    cat_cnt = np.zeros(NUM_CLASSES).astype(np.int32)
    res_mask = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], NUM_CLASSES)).astype(np.int32)
    for ann in annotations:
        # cat = ann.category.name.lower().replace('_', ' ')
        cat = ann['categories'][0]['name'].lower().replace('_', ' ')
        if cat not in CATEGORIES_IDS:
            continue
        cat_i = CATEGORIES_IDS[cat]
        res_mask[..., cat_i] |= _convert_annotation_to_mask(ann)
        cat_cnt[cat_i] += 1

    return res_mask, cat_cnt



