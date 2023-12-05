import os
from typing import Optional, List
from functools import lru_cache
from kili.client import Kili
from kili.services.label_data_parsing.annotation import BoundingPolyAnnotation
import numpy as np

from cs_sem_seg.configs import KILI_PROJECT_ID


@lru_cache()
def _connect_to_kili() -> Kili:
    return Kili(api_key=os.environ['AUTH_SECRET'])

def _download(kili_external_id: str, local_file_dir: Optional[str] = None) -> str:

    filename = kili_external_id + ".png"

    # if local_file_path is not specified saving in home dir
    if local_file_dir is None:
        local_file_dir = os.path.join("~/.cache/kili/projects", KILI_PROJECT_ID, "assets", filename)

    local_file_path = os.path.join(local_file_dir, filename)

    # check if file already exists
    if os.path.exists(local_file_path):
        return local_file_path

    kili = _connect_to_kili()

    kili.assets(
        project_id=KILI_PROJECT_ID,
        external_id_strictly_in=kili_external_id,
        download_media=True,
        local_media_dir=local_file_dir,
        fields=[]
        )

    return local_file_path

def _convert_annotation_to_np(annotation: BoundingPolyAnnotation):
    vertices = annotation.bounding_poly[0].normalized_vertices
    return np.array([[vertex['x'], vertex['y']] for vertex in vertices])

def _get_masks(kili_external_id: str):

    kili = _connect_to_kili()

    labels = kili.labels(
        project_id=KILI_PROJECT_ID,
        asset_external_id_in=[kili_external_id],
        fields=['jsonResponse'],
        output_format='parsed_label'
    )
    annotations = labels[0].jobs["OBJECT_DETECTION_JOB"].annotations
    masks_list: List[np.ndarray] = []
    category_list: List[str] = []

    for ann in annotations:
        masks_list.append(_convert_annotation_to_np(ann))

        mask_category = ann.category.name
        category_list.append(mask_category)



    return masks_list, category_list
