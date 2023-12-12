from typing import List
from code_loader.contract.datasetclasses import PreprocessResponse

# from cs_sem_seg.data.cs_data import get_cityscapes_data
from cs_sem_seg.configs import TRAIN_SIZE, VAL_SIZE, TEST_SIZE
from cs_sem_seg.utils.kili_utils import get_killi_assets


def load_data() -> List[PreprocessResponse]:
    # cs_responses: List[PreprocessResponse] = get_cityscapes_data()
    responses = []
    sub_sizes = [TRAIN_SIZE, VAL_SIZE]
    sub_names = ["train", "val"]
    kili_assets, kili = get_killi_assets(sub_names, sub_sizes)
    for i, title in enumerate(sub_names):
        responses += [PreprocessResponse(data=dict(data=kili_assets[i], kili=kili), length=min(sub_sizes[i], len(kili_assets[i])))]
    return responses


def load_test_data() -> PreprocessResponse:
    sub_size = [TEST_SIZE]
    sub_names = ["test"]
    kili_assets, kili = get_killi_assets(sub_names, sub_size)
    responses = PreprocessResponse(data=dict(data=kili_assets[0], kili=kili), length=min(sub_size[0], len(kili_assets)))
    return responses

