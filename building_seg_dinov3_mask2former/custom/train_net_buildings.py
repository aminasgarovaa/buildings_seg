import os
import glob
from typing import List, Dict

import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import SemSegEvaluator
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler

from mask2former import add_maskformer2_config

from custom.add_dinov3_config import add_dinov3_config
from building_seg_dinov3_mask2former.custom.npy_semseg_mapper import NpyMaskFormerSemanticDatasetMapper

# make sure backbone is registered
import custom.dinov3_fpn_backbone  # noqa: F401


def register_building_semseg(
    name: str,
    images_dir: str,
    masks_dir: str,
):
    def _loader() -> List[Dict]:
        img_paths = sorted(glob.glob(os.path.join(images_dir, "*.npy")))
        items = []
        for p in img_paths:
            base = os.path.splitext(os.path.basename(p))[0]
            m = os.path.join(masks_dir, base + ".npy")
            if not os.path.exists(m):
                continue

            # For sem seg, Detectron2 expects sem_seg_file_name. :contentReference[oaicite:9]{index=9}
            items.append({
                "file_name": p,
                "sem_seg_file_name": m,
                "image_id": base,
            })
        return items

    DatasetCatalog.register(name, _loader)
    meta = MetadataCatalog.get(name)
    meta.evaluator_type = "sem_seg"
    meta.stuff_classes = ["background", "building"]


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic_npy":
            mapper = NpyMaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        return super().build_train_loader(cfg)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_dinov3_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        Trainer.test(cfg, model)
        return

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
