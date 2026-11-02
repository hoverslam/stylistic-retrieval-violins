from __future__ import annotations

import warnings
from pathlib import Path

import torch
import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps
import matplotlib.pyplot as plt

warnings.simplefilter("ignore", PILImage.DecompressionBombWarning)


def load_image(img_path: Path | str, color_mode: str) -> np.ndarray:
    """
    Loads image from path and returns as a 3-channel NumPy array.

    Args:
        img_path: Path or str to the image file.
        color_mode: 'RGB' or 'GRAY' (case-insensitive).

    Returns:
        np.ndarray: Image array of shape (H, W, 3) and dtype uint8.
    """
    img = PILImage.open(Path(img_path).as_posix())
    img = ImageOps.exif_transpose(img)

    mode = color_mode.upper()
    if mode == "RGB":
        img = img.convert("RGB")
        return np.array(img)

    elif mode == "GRAY":
        img = img.convert("L")
        np_img = np.array(img)
        return np.stack([np_img, np_img, np_img], axis=-1)

    else:
        raise ValueError(f"Unsupported color_mode: {color_mode}.")


def show_image(
    image: torch.Tensor,
    label: str = "",
    dim_order: str = "CHW",
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> None:
    """
    Displays a torch.Tensor as an image, handling denormalization and dimension ordering.

    Args:
        image (torch.Tensor): The image tensor to display.
        label (str): Title for the plot.
        dim_order (str): The order of dimensions in the input 'image' tensor.
                         Options: "CHW" (standard PyTorch) or "HWC" (standard NumPy/OpenCV).
        mean (list[float], optional): Mean used for normalization to reverse it.
        std (list[float], optional): Std used for normalization to reverse it.
    """
    img = image.clone().detach()

    # Standardize to HWC immediately
    if dim_order.upper() == "CHW":
        img = img.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]

    # Denormalize
    if mean is not None and std is not None:
        # view(1, 1, 3) aligns the 3 channels with the last dimension of [H, W, 3]
        mean_t = torch.tensor(mean, device=img.device).view(1, 1, 3)
        std_t = torch.tensor(std, device=img.device).view(1, 1, 3)
        img = img * std_t + mean_t

    # Render
    plt.figure(figsize=(4, 4))
    plt.imshow(img.cpu().numpy())
    plt.title(label)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_image_grid(
    images: list[torch.Tensor],
    labels: list[str] | None,
    ncols: int = 5,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> None:
    nrows = (len(images) + ncols - 1) // ncols
    _, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()

    for i, img in enumerate(images):
        img = img.clone()

        # Denormalize if mean and std are provided
        if mean is not None and std is not None:
            mean_t = torch.tensor(mean, device=img.device).view(3, 1, 1)
            std_t = torch.tensor(std, device=img.device).view(3, 1, 1)
            img = img * std_t + mean_t

        img = img.clamp(0, 1)

        axes[i].imshow(img.permute(1, 2, 0).cpu().numpy())
        if labels is not None and labels[i]:
            axes[i].set_title(labels[i])
        axes[i].axis("off")

    for i in range(len(images), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


class Image:
    def __init__(self, path: Path | str, aois: list[dict] | None = None) -> None:
        self.path = Path(path)
        self.aois = aois if aois is not None else []

    @property
    def name(self) -> str:
        return self.path.name

    def __repr__(self) -> str:
        out = f"Image(path='{self.path.as_posix()}'"

        if self.aois:
            out += f", aois={set([aoi['label'] for aoi in self.aois])}"

        out += ")"

        return out

    def numpy(self, color_mode: str = "RGB") -> np.ndarray:
        """
        Loads image and returns as a NumPy array in H x W x C format.

        Args:
            color_mode: 'RGB' or 'GRAY' (case-insensitive).

        Returns:
            np.ndarray: Image array of shape (H, W, 3) and dtype uint8.
        """
        return load_image(self.path, color_mode)

    def tensor(self, color_mode: str = "RGB") -> torch.Tensor:
        """
        Loads image and returns as a PyTorch Tensor in C x H x W format.

        Args:
            color_mode: 'RGB' or 'GRAY' (case-insensitive).

        Returns:
            torch.Tensor: Tensor of shape (3, H, W) and dtype uint8.
        """
        img_np = load_image(self.path, color_mode)

        return torch.from_numpy(img_np.transpose(2, 0, 1))
