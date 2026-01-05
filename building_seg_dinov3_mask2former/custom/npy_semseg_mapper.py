import copy
import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

class NpyMaskFormerSemanticDatasetMapper:
    def __init__(self, cfg, is_train: bool = True):
        self.is_train = is_train
        self.img_format = cfg.INPUT.FORMAT  # "RGB" in base configs :contentReference[oaicite:7]{index=7}

        if is_train:
            self.augmentations = [
                T.RandomFlip(horizontal=True, vertical=False),
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                ),
            ]
            self.crop_gen = None
            if cfg.INPUT.CROP.ENABLED:
                self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
        else:
            self.augmentations = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, "choice"
                )
            ]
            self.crop_gen = None

    def _load_image_npy(self, path: str) -> np.ndarray:
        arr = np.load(path)
        # Support CHW or HWC
        if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"Bad image shape {arr.shape} for {path}")

        arr = arr[:, :, :3]

        # If uint16 but looks like 0..255, cast to uint8
        if arr.dtype == np.uint16 and arr.max() <= 255:
            arr = arr.astype(np.uint8)

        # If float in 0..1, convert to 0..255
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.5:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

        # Otherwise assume it is already 0..255
        if arr.dtype != np.uint8:
            arr = arr.clip(0, 255).astype(np.uint8)

        return arr

    def _load_mask_npy(self, path: str) -> np.ndarray:
        m = np.load(path)
        if m.ndim == 3:
            # if mask saved as (1,H,W) or (H,W,1)
            if m.shape[0] == 1:
                m = m[0]
            elif m.shape[-1] == 1:
                m = m[..., 0]
        if m.ndim != 2:
            raise ValueError(f"Bad mask shape {m.shape} for {path}")
        return m.astype(np.int64)

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        image = self._load_image_npy(dataset_dict["file_name"])
        sem_seg = self._load_mask_npy(dataset_dict["sem_seg_file_name"])

        aug_input = T.AugInput(image, sem_seg=sem_seg)
        transforms = T.AugmentationList(self.augmentations)(aug_input)

        image = aug_input.image
        sem_seg = aug_input.sem_seg

        if self.crop_gen is not None:
            crop_tfm = self.crop_gen.get_transform(image)
            image = crop_tfm.apply_image(image)
            sem_seg = crop_tfm.apply_segmentation(sem_seg)

        image = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        sem_seg = torch.as_tensor(sem_seg.astype("int64"))

        dataset_dict["image"] = image
        dataset_dict["sem_seg"] = sem_seg
        return dataset_dict
