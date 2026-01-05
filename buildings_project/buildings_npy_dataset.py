import os
import glob
from detectron2.data import DatasetCatalog, MetadataCatalog

def _pair_npy(image_dir, mask_dir):
    # pairs by filename without extension
    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.npy")))
    items = []
    for p in img_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        m = os.path.join(mask_dir, base + ".npy")
        if not os.path.exists(m):
            raise FileNotFoundError(f"Mask not found for {p}: expected {m}")
        items.append((p, m))
    return items

def get_buildings_semseg_dicts(image_dir, mask_dir):
    pairs = _pair_npy(image_dir, mask_dir)
    dataset_dicts = []
    for idx, (img_path, mask_path) in enumerate(pairs):
        dataset_dicts.append({
            "file_name": img_path,
            "sem_seg_file_name": mask_path,
            "image_id": idx,
        })
    return dataset_dicts

def register_buildings_datasets(root_dir):
    train_name = "buildings_sem_seg_train"
    val_name = "buildings_sem_seg_val"

    train_img = os.path.join(root_dir, "train_images")
    train_msk = os.path.join(root_dir, "train_masks")
    val_img = os.path.join(root_dir, "val_images")
    val_msk = os.path.join(root_dir, "val_masks")

    DatasetCatalog.register(train_name, lambda: get_buildings_semseg_dicts(train_img, train_msk))
    DatasetCatalog.register(val_name, lambda: get_buildings_semseg_dicts(val_img, val_msk))

    meta = {
        "stuff_classes": ["background", "building"],
        "evaluator_type": "sem_seg",
        "ignore_label": 255,
    }
    MetadataCatalog.get(train_name).set(**meta)
    MetadataCatalog.get(val_name).set(**meta)
