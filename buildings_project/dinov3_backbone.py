import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

@BACKBONE_REGISTRY.register()
class Dinov3Backbone(Backbone):
    """
    Outputs a dict with:
      res2: stride 4
      res3: stride 8
      res4: stride 16
      res5: stride 32
    """

    def __init__(self, cfg, input_shape):
        super().__init__()
        repo = cfg.MODEL.DINOV3.REPO
        weights = cfg.MODEL.DINOV3.WEIGHTS
        model_name = cfg.MODEL.DINOV3.MODEL_NAME
        out_ch = cfg.MODEL.DINOV3.OUT_CHANNELS
        freeze = cfg.MODEL.DINOV3.FREEZE

        self.vit = torch.hub.load(repo, model_name, source="local", weights=weights)
        self.vit.eval()

        # DINOv3 ViT-L typically has embed dim 1024
        vit_dim = cfg.MODEL.DINOV3.VIT_DIM

        # Simple pyramid from stride16 map
        self.to_res4 = nn.Conv2d(vit_dim, out_ch, kernel_size=1)

        self.to_res3 = nn.Sequential(
            nn.Conv2d(vit_dim, out_ch, kernel_size=1),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.to_res2 = nn.Sequential(
            nn.Conv2d(vit_dim, out_ch, kernel_size=1),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.to_res5 = nn.Sequential(
            nn.Conv2d(vit_dim, out_ch, kernel_size=1),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1),
        )

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_feature_channels = {k: out_ch for k in self._out_features}

    def forward(self, x):
        """
        x: tensor BCHW, already normalized by Detectron2 preprocess using PIXEL_MEAN / PIXEL_STD.
        """
        # Get stride16 feature map from last layer
        feats = self.vit.get_intermediate_layers(
            x, n=1, reshape=True, return_class_token=False, norm=True
        )[0]

        # Make robust if it comes as BHWC
        if feats.ndim == 4 and feats.shape[1] != x.shape[2] // 16:
            # likely BCHW already, do nothing
            pass
        if feats.ndim == 4 and feats.shape[-1] == feats.shape[-2]:
            pass

        # If DINO returns BHWC, convert to BCHW
        if feats.ndim == 4 and feats.shape[1] != self._infer_channels(feats):
            # heuristic, skip
            pass

        if feats.ndim == 3:
            raise ValueError("Got tokens shape (B, HW, C). reshape=True should return a feature map.")

        # If feats is BHWC
        if feats.shape[1] != x.shape[1] and feats.shape[-1] != x.shape[-1] and feats.shape[-1] != feats.shape[1]:
            # not reliable, keep as is
            pass

        if feats.shape[1] != self.to_res4.in_channels and feats.shape[-1] == self.to_res4.in_channels:
            feats = feats.permute(0, 3, 1, 2).contiguous()

        # feats is now BCHW at stride16
        res4 = self.to_res4(feats)

        # stride8 and stride4 by upsampling the stride16 map
        f8 = F.interpolate(feats, scale_factor=2.0, mode="bilinear", align_corners=False)
        f4 = F.interpolate(feats, scale_factor=4.0, mode="bilinear", align_corners=False)

        res3 = self.to_res3(f8)
        res2 = self.to_res2(f4)

        # stride32 by downsampling from stride16
        res5 = self.to_res5(feats)

        return {"res2": res2, "res3": res3, "res4": res4, "res5": res5}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    @staticmethod
    def _infer_channels(feats):
        # helper for heuristics
        return feats.shape[1]
