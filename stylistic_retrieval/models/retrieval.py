from __future__ import annotations

from stylistic_retrieval.models.backbone import ResNet, DinoV2, DINOV2_DIMS
from stylistic_retrieval.models.aggregation import GAP, GeM, NetVLAD, SemanticSpatialFusion
from stylistic_retrieval.models.cricavpr import CricaVPRRetrieval
from stylistic_retrieval.utils.config import Config

import torch
from torch import nn
import torch.nn.functional as F


def setup_retrieval_model(config: Config) -> nn.Module:
    model_config = config.as_dict["model"]
    model_params = model_config.get("params", {})

    match model_config["architecture"]:
        case "ResNet":
            return ResNetRetrieval(**model_params)
        case "DinoV2":
            return DinoV2Retrieval(**model_params)
        case "CricaVPR":
            return CricaVPRRetrieval(**model_params)
        case _:
            raise ValueError(f"Unknown architecture '{model_config['architecture']}'")


class ResNetRetrieval(nn.Module):
    def __init__(
        self,
        num_layers: int,
        aggregation: str,
        agg_params: dict | None = None,
        pretrained: bool = True,
        projection_dim: int | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.aggregation_type = aggregation
        self.agg_params = agg_params if agg_params is not None else {}

        # Initialize Backbone with freeze toggle
        self._backbone = ResNet(num_layers, pretrained, freeze=freeze_backbone)
        dim = 512 if num_layers < 50 else 2048

        # Setup Aggregation
        self._pooling = self._setup_aggregation(self.aggregation_type, dim, self.agg_params)

        # Optional Projection Layer
        self._projection = nn.Linear(dim, projection_dim) if projection_dim else nn.Identity()

    def _setup_aggregation(self, name: str, dim: int, params: dict) -> nn.Module:
        match name:
            case "GAP":
                return GAP()
            case "GeM":
                return GeM(**params)
            case "NetVLAD":
                return NetVLAD(dim=dim, **params)
            case _:
                raise ValueError(f"Unknown aggregation method: '{name}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._backbone(x)

        if self.aggregation_type == "NetVLAD":
            embedding = self._pooling(x)
        else:
            embedding = self._pooling(x).flatten(1)

        # Apply optional projection and final normalization
        embedding = self._projection(embedding)

        return F.normalize(embedding, dim=1)


class DinoV2Retrieval(nn.Module):
    def __init__(
        self,
        model_name: str,
        aggregation: str,
        agg_params: dict | None = None,
        lora: dict | None = None,
        pretrained: bool = True,
        projection_dim: int | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.aggregation_type = aggregation
        self.agg_params = agg_params if agg_params is not None else {}

        # Initialize Backbone with freeze toggle
        self._backbone = DinoV2(model_name, pretrained, lora_config=lora, freeze=freeze_backbone)
        dim = DINOV2_DIMS[model_name]

        # Setup Aggregation
        if self.aggregation_type != "CLS":
            self._pooling = self._setup_aggregation(self.aggregation_type, dim, self.agg_params)

        # Optional Projection Layer
        self._projection = nn.Linear(dim, projection_dim) if projection_dim else nn.Identity()

    def _setup_aggregation(self, name: str, dim: int, params: dict) -> nn.Module:
        match name:
            case "GAP":
                return GAP()
            case "GeM":
                return GeM(**params)
            case "NetVLAD":
                return NetVLAD(dim=dim, **params)
            case "SemanticSpatialFusion":
                return SemanticSpatialFusion(dim=dim, **params)
            case _:
                raise ValueError(f"Unknown aggregation method: '{name}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._backbone(x)
        cls_token = features["x_norm_clstoken"]
        patch_tokens = features["x_norm_patchtokens"]

        # Aggregation logic
        if self.aggregation_type == "CLS":
            embedding = cls_token
        elif self.aggregation_type == "SemanticSpatialFusion":
            embedding = self._pooling(cls_token, patch_tokens)
        else:
            b, n, c = patch_tokens.shape
            h = w = int(n**0.5)
            patch_tokens = patch_tokens.permute(0, 2, 1).reshape(b, c, h, w)
            embedding = self._pooling(patch_tokens).flatten(1)

        # Apply optional projection and final normalization
        embedding = self._projection(embedding)

        return F.normalize(embedding, dim=1)
