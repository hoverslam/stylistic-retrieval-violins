# Modified version of CricaVPR: https://github.com/Lu-Feng/CricaVPR

from __future__ import annotations

from stylistic_retrieval.models.cricavpr.network import CricaVPRNet

import torch
from torch import nn


class CricaVPRRetrieval(nn.Module):
    """Wraps the original CricaVPRNet (DINOv2 ViT/14 + MulConvAdapters + cross-image encoder).

    Note: the cross-image encoder attends across all images in a batch, so embeddings depend
    on the batch composition. Use `test_settings.batch_by_instrument: true` for evaluation.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        pretrained: bool = True,
        weights_path: str | None = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.net = CricaVPRNet(pretrained_foundation=False, model_name=model_name)

        if pretrained:
            self._load_foundation_weights(weights_path)

        # Freeze everything except adapter (from the original train.py)
        if freeze_backbone:
            for name, param in self.net.backbone.named_parameters():
                if "adapter" not in name:
                    param.requires_grad = False

        # Init adapter layers (from the original train.py)
        for n, m in self.named_modules():
            if "adapter" in n:
                for n2, m2 in m.named_modules():
                    if isinstance(m2, nn.Linear) and "D_fc2" in n2:
                        nn.init.constant_(m2.weight, 0.0)
                        nn.init.constant_(m2.bias, 0.0)

                    if isinstance(m2, nn.Conv2d):
                        nn.init.constant_(m2.weight, 1e-5)
                        nn.init.constant_(m2.bias, 1e-5)

    def _load_foundation_weights(self, weights_path: str | None) -> None:
        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location="cpu")
        else:
            state_dict = torch.hub.load("facebookresearch/dinov2", self.model_name, pretrained=True).state_dict()

        # Same update-and-load logic as the original get_backbone()
        model_dict = self.net.backbone.state_dict()
        model_dict.update(state_dict.items())
        self.net.backbone.load_state_dict(model_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
