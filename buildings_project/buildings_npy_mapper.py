import numpy as np
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

def _load_image_npy(path):
    img = np.load(path)
    # allow CHW or HWC
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))

    if img.dtype != np.uint8:
        # if float in [0..1], scale to [0..255]
        img = img.astype(np.float32)
        if img.max() <= 1.5:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img  # HWC uint8 RGB expected later

def _load_mask_npy(path):
    m = np.load(path)
    if m.ndim != 2:
        raise ValueError(f"Mask must be (H,W). Got {m.shape} from {path}")
    # enforce 0/1
    m = (m > 0).astype(np.uint8)
    return m  # H,W

def _pad_to_divisibility(img, mask, divisibility):
    if divisibility <= 0:
        return img, mask
    h, w = img.shape[:2]
    pad_h = (divisibility - (h % divisibility)) % divisibility
    pad_w = (divisibility - (w % divisibility)) % divisibility
    if pad_h == 0 and pad_w == 0:
        return img, mask

    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=255)
    return img, mask

class BuildingsNPYSemanticMapper:
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.img_format = cfg.INPUT.FORMAT  # we will set "RGB"
        self.size_div = cfg.INPUT.SIZE_DIVISIBILITY

        if is_train:
            self.augmentations = [
                T.RandomFlip(horizontal=True, vertical=False),
            ]
        else:
            self.augmentations = []

    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()

        image = _load_image_npy(dataset_dict["file_name"])
        sem_seg = _load_mask_npy(dataset_dict["sem_seg_file_name"])

        aug_input = T.AugInput(image, sem_seg=sem_seg)
        transforms = T.AugmentationList(self.augmentations)(aug_input)

        image = aug_input.image
        sem_seg = aug_input.sem_seg

        image, sem_seg = _pad_to_divisibility(image, sem_seg, self.size_div)

        # Detectron2 expects CHW float32 tensor
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        sem_seg = torch.as_tensor(sem_seg.astype("int64"))

        dataset_dict["image"] = image
        dataset_dict["sem_seg"] = sem_seg
        return dataset_dict
