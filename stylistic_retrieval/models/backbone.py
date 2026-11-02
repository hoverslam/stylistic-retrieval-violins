from peft import get_peft_model, LoraConfig
import torch
from torch import nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    resnet34,
    ResNet34_Weights,
    resnet50,
    ResNet50_Weights,
    resnet101,
    ResNet101_Weights,
    resnet152,
    ResNet152_Weights,
)

DINOV2_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class DinoV2(nn.Module):
    def __init__(
        self, model_name: str, pretrained: bool, lora_config: dict | None = None, freeze: bool = False
    ) -> None:
        super().__init__()
        if model_name not in DINOV2_DIMS:
            raise ValueError(f"Unknown DINOv2 model name: '{model_name}'")

        self.model_name = model_name
        backbone = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=pretrained)

        if lora_config is not None:
            self._backbone = get_peft_model(backbone, peft_config=LoraConfig(**lora_config))  # type: ignore
            self._backbone.print_trainable_parameters()
        else:
            self._backbone = backbone
            if freeze:
                for param in self._backbone.parameters():  # type: ignore
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Correctly unwrap PEFT model if it exists to access forward_features
        model = self._backbone.base_model.model if hasattr(self._backbone, "base_model") else self._backbone  # type: ignore
        return model.forward_features(x)  # type: ignore

    def get_self_attention(self, x: torch.Tensor) -> torch.Tensor:
        # Unwrap for manual attention calculation
        base_model = self._backbone.base_model.model if hasattr(self._backbone, "base_model") else self._backbone  # type: ignore

        x = base_model.prepare_tokens_with_masks(x)  # type: ignore
        for blk in base_model.blocks[:-1]:  # type: ignore
            x = blk(x)

        blk = base_model.blocks[-1]  # type: ignore
        x_norm = blk.norm1(x)
        attn_layer = blk.attn

        qkv = attn_layer.qkv(x_norm)
        B, N, C = x_norm.shape
        qkv = qkv.reshape(B, N, 3, attn_layer.num_heads, C // attn_layer.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]

        attn = (q @ k.transpose(-2, -1)) * attn_layer.scale
        return attn.softmax(dim=-1)

    def __repr__(self):
        return f"DinoV2(model_name='{self.model_name}')"


class ResNet(nn.Module):
    def __init__(self, num_layers: int, pretrained: bool, freeze: bool = False) -> None:
        super().__init__()
        if num_layers == 18:
            resnet = resnet18(weights=ResNet18_Weights.DEFAULT) if pretrained else resnet18()
        elif num_layers == 34:
            resnet = resnet34(weights=ResNet34_Weights.DEFAULT) if pretrained else resnet34()
        elif num_layers == 50:
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT) if pretrained else resnet50()
        elif num_layers == 101:
            resnet = resnet101(weights=ResNet101_Weights.DEFAULT) if pretrained else resnet101()
        elif num_layers == 152:
            resnet = resnet152(weights=ResNet152_Weights.DEFAULT) if pretrained else resnet152()
        else:
            raise ValueError("Unknown number of layers. Expect: '18', '34', '50', '101', or '152'.")

        self._backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        if freeze:
            for param in self._backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(x)

    def __repr__(self):
        return self._backbone.__repr__()
