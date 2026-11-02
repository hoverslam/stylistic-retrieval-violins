import torch
from torch import nn
import torch.nn.functional as F


class GAP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._pooling(x)


class GeM(nn.Module):
    def __init__(self, p: float, eps: float = 1e-6):
        super().__init__()
        self._p = nn.Parameter(torch.ones(1) * p)
        self._eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x.clamp(min=self._eps).pow(self._p), (1, 1)).pow(1.0 / self._p)

    def __repr__(self):
        return f"GeM(p={self._p.data.tolist()[0]:.4f}, eps={str(self._eps)})"


class SemanticSpatialFusion(nn.Module):
    """Aggregation that fuses global semantic and local spatial information.

    This module concatenates the [CLS] token (global semantics) with a
    pooled representation of the patch tokens (spatial details) and
    projects the result into a joint embedding space. Research indicates
    this dual-token approach provides a richer representation for
    retrieval tasks.

    Attributes:
        patch_pooling: The pooling layer used to aggregate patch tokens.
        projection: Linear layer followed by LayerNorm to project fused features.
    """

    def __init__(self, dim: int, embedding_dim: int | None = None) -> None:
        """Initializes the SemanticSpatialFusion module.

        Args:
            dim: The input dimension of the backbone features.
            embedding_dim: Optional dimension for the final output vector.
                Defaults to the input dim if None.
        """
        super().__init__()
        self.patch_pooling = GAP()

        # Output dimension defaults to the backbone dimension if not specified
        output_dim = embedding_dim if embedding_dim is not None else dim

        # Input is dim * 2 due to [CLS; PooledPatches] concatenation
        self.projection = nn.Sequential(nn.Linear(dim * 2, output_dim), nn.LayerNorm(output_dim))

    def forward(self, cls_token: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Fuses global and spatial tokens into a single embedding.

        Args:
            cls_token: The [CLS] token tensor of shape (B, C).
            patch_tokens: The patch token tensor of shape (B, N, C).

        Returns:
            A projected and fused embedding tensor of shape (B, output_dim).
        """
        # Reshape patches: (B, N, C) -> (B, C, H, W) for 2D pooling
        b, n, c = patch_tokens.shape
        h = w = int(n**0.5)
        patches = patch_tokens.permute(0, 2, 1).reshape(b, c, h, w)

        # Pool patches to (B, C)
        pooled_patches = self.patch_pooling(patches).flatten(1)

        # Concatenate [CLS] and Patches: (B, 2*C)
        combined = torch.cat([cls_token, pooled_patches], dim=1)

        # Project to final joint embedding space
        return self.projection(combined)


# ! FROM: https://github.com/lyakaap/NetVLAD-pytorch
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters: int, dim: int, alpha: float, normalize_input=True) -> None:
        """
        Args:
            num_clusters (int): The number of clusters.
            dim (int): Dimension of descriptors.
            alpha (float): Parameter of initialization. Larger value is harder assignment.
            normalize_input (bool): If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter((2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1))
        self.conv.bias = nn.Parameter(-self.alpha * self.centroids.norm(dim=1))

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - self.centroids.expand(
            x_flatten.size(-1), -1, -1
        ).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad
