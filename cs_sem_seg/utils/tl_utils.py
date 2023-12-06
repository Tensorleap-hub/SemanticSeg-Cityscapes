from typing import List, Tuple
from code_loader.contract.datasetclasses import PreprocessResponse

# from cs_sem_seg.data.cs_data import get_cityscapes_data
from cs_sem_seg.configs import TRAIN_SIZE, VAL_SIZE, TEST_SIZE
from cs_sem_seg.utils.kili_utils import get_killi_assets



def subset_images() -> List[PreprocessResponse]:
    # cs_responses: List[PreprocessResponse] = get_cityscapes_data()
    responses = []
    subset_sizes = [TRAIN_SIZE, VAL_SIZE, TEST_SIZE]
    kili_assets = get_killi_assets()
    sub_names = ["train", "val", "test"]
    for i, title in enumerate(sub_names):
        responses += [PreprocessResponse(data=kili_assets[i], length=min(subset_sizes[i], len(kili_assets[i])))]
    return responses

