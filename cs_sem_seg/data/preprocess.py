from typing import List, Dict
from code_loader.contract.datasetclasses import PreprocessResponse

from cs_sem_seg.data.cs_data import get_cityscapes_data
from cs_sem_seg.configs import TRAIN_SIZE, VAL_SIZE


def subset_images() -> List[PreprocessResponse]:
    subset_sizes = [TRAIN_SIZE, VAL_SIZE]
    cs_responses: List[PreprocessResponse] = get_cityscapes_data()
    sub_names = ["train", "validation"]
    for i, title in enumerate(sub_names):
        cs_responses[i].length = min(subset_sizes[i], cs_responses[i].length)
    return cs_responses