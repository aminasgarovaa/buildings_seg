import os
import sys
import torch

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer

# Mask2Former config function
from mask2former import add_maskformer2_config

from buildings_npy_dataset import register_buildings_datasets
from buildings_npy_mapper import BuildingsNPYSemanticMapper

# Make sure our backbone file is imported so it registers in BACKBONE_REGISTRY
import dinov3_backbone  # noqa: F401

class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.DATASET_MAPPER_NAME == "buildings_npy_semantic":
            mapper = BuildingsNPYSemanticMapper(cfg, is_train=True)
            return build_detection_train_loader(cfg, mapper=mapper)
        return super().build_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if cfg.INPUT.DATASET_MAPPER_NAME == "buildings_npy_semantic":
            mapper = BuildingsNPYSemanticMapper(cfg, is_train=False)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        return super().build_test_loader(cfg, dataset_name)

def setup(args):
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    # register datasets
    register_buildings_datasets(os.path.join(os.path.dirname(__file__), "data"))

    cfg = setup(args)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer.train()

if __name__ == "__main__":
    from detectron2.engine import default_argument_parser
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
