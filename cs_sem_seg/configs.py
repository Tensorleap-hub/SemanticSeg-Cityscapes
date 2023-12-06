import os
import numpy as np
# --------------- GCS --------------------
BUCKET_NAME = 'label-backend-production'
# PROJECT_ID = 'tl-private-dev-project'

# --------------- Kili --------------------
KILI_PROJECT_ID = "clpbb902w01bg087p9o6l0691"

# --------------- Data --------------------
LOCAL_DIR = os.path.join("~/.cache/kili/projects", KILI_PROJECT_ID, "assets")       # todo: change to nfs
NORM_CS = False
SEED = 42
NUM_CLASSES = 19
IMAGE_SIZE = (2048, 1024)
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 1000, 500, 500
TRAIN_PERCENT = 0.8

SUPERCATEGORY_GROUNDTRUTH = False

LOAD_UNION_CATEGORIES_IMAGES = False

# --------------- Augmentations --------------------
APPLY_AUGMENTATION = True
IMAGE_MEAN = np.array([0.485, 0.456, 0.406])
IMAGE_STD = np.array([0.229, 0.224, 0.225])
CITYSCAPES_MEAN = np.array([0.287, 0.325, 0.284])
CITYSCAPES_STD = np.array([0.176, 0.181, 0.178])
VAL_INDICES = [190, 198, 45, 25, 141, 104, 17, 162, 49, 167, 168, 34, 150, 113, 44,
               182, 196, 11, 6, 46, 133, 74, 81, 65, 66, 79, 96, 92, 178, 103]
AUGMENT = True
SUBSET_REPEATS = [1, 1]

# Augmentation limits
HUE_LIM = 0.3 / np.pi
SATUR_LIM = 0.3
BRIGHT_LIM = 0.3
CONTR_LIM = 0.3
