import torch
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, get_norm
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

def _infer_embed_dim(dino_model) -> int:
    for attr in ["embed_dim", "dim", "hidden_dim"]:
        if hasattr(dino_model, attr):
            v = getattr(dino_model, attr)
            if isinstance(v, int):
                return v
    if hasattr(dino_model, "norm") and hasattr(dino_model.norm, "weight"):
        return int(dino_model.norm.weight.shape[0])
    raise ValueError("Could not infer DINOv3 embed dim")

@BACKBONE_REGISTRY.register()
class DinoV3FPNBackbone(Backbone):
    @configurable
    def __init__(
        self,
        *,
        repo_dir: str,
        weights_path: str,
        model_name: str,
        out_channels: int,
        freeze: bool,
        norm: str = "GN",
    ):
        super().__init__()

        self.dino = torch.hub.load(
            repo_dir,
            model_name,
            source="local",
            weights=weights_path,
        )

        if freeze:
            for p in self.dino.parameters():
                p.requires_grad = False
            self.dino.eval()

        self.embed_dim = _infer_embed_dim(self.dino)
        self.out_channels = out_channels

        self.lateral2 = Conv2d(self.embed_dim, out_channels, kernel_size=1, norm=get_norm(norm, out_channels))
        self.lateral3 = Conv2d(self.embed_dim, out_channels, kernel_size=1, norm=get_norm(norm, out_channels))
        self.lateral4 = Conv2d(self.embed_dim, out_channels, kernel_size=1, norm=get_norm(norm, out_channels))
        self.lateral5 = Conv2d(self.embed_dim, out_channels, kernel_size=1, norm=get_norm(norm, out_channels))

        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_feature_channels = {k: out_channels for k in self._out_feature_strides.keys()}
        self._out_features = list(self._out_feature_strides.keys())

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "repo_dir": cfg.MODEL.DINOV3.REPO_DIR,
            "weights_path": cfg.MODEL.DINOV3.WEIGHTS_PATH,
            "model_name": cfg.MODEL.DINOV3.MODEL_NAME,
            "out_channels": cfg.MODEL.DINOV3.OUT_CHANNELS,
            "freeze": cfg.MODEL.DINOV3.FREEZE,
            "norm": "GN",
        }

    def forward(self, x):
        # x is already normalized by Mask2Former using cfg.MODEL.PIXEL_MEAN / PIXEL_STD. :contentReference[oaicite:5]{index=5}
        with torch.set_grad_enabled(self.dino.training):
            feats = self.dino.get_intermediate_layers(
                x, n=1, reshape=True, return_class_token=False, norm=True
            )
            # the issue page shows this signature for get_intermediate_layers. :contentReference[oaicite:6]{index=6}
            feat16 = feats[0]  # (B, C, H/16, W/16)

        # Build res2..res5 from stride 16 feature
        res4 = feat16
        res3 = F.interpolate(feat16, scale_factor=2.0, mode="bilinear", align_corners=False)
        res2 = F.interpolate(feat16, scale_factor=4.0, mode="bilinear", align_corners=False)
        res5 = F.max_pool2d(feat16, kernel_size=2, stride=2)

        out = {
            "res2": self.lateral2(res2),
            "res3": self.lateral3(res3),
            "res4": self.lateral4(res4),
            "res5": self.lateral5(res5),
        }
        return out

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32
