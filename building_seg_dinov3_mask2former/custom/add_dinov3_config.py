from detectron2.config import CfgNode as CN

def add_dinov3_config(cfg):
    cfg.MODEL.DINOV3 = CN()
    cfg.MODEL.DINOV3.REPO_DIR = ""
    cfg.MODEL.DINOV3.WEIGHTS_PATH = ""
    cfg.MODEL.DINOV3.MODEL_NAME = "dinov3_vitl16"
    cfg.MODEL.DINOV3.OUT_CHANNELS = 256
    cfg.MODEL.DINOV3.FREEZE = True
