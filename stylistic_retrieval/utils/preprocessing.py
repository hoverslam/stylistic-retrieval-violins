from stylistic_retrieval.utils.config import Config

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def setup_transform(config: Config, mode: str) -> A.Compose:
    match mode:
        case "train":
            transform_config = config.as_dict["train_transforms"]
        case "test":
            transform_config = config.as_dict["test_transforms"]
        case _:
            raise ValueError(f"Invalid mode: '{mode}'. Expected 'train' or 'test'.")

    aug_list = []

    # (1) Spatial: Resize
    if "random_resized_crop" in transform_config:
        aug_list.append(A.RandomResizedCrop(**transform_config["random_resized_crop"], interpolation=cv2.INTER_CUBIC))

    if "resize" in transform_config:
        aug_list.append(A.LongestMaxSize(**transform_config["resize"], interpolation=cv2.INTER_CUBIC))

    # (2) Square / Padding / Divisibility
    if "pad_if_needed" in transform_config:
        aug_list.append(A.PadIfNeeded(**transform_config["pad_if_needed"], border_mode=cv2.BORDER_CONSTANT))

    # (3) Geometric
    if "random_horizontal_flip" in transform_config:
        aug_list.append(A.HorizontalFlip(**transform_config["random_horizontal_flip"]))

    if "random_vertical_flip" in transform_config:
        aug_list.append(A.VerticalFlip(**transform_config["random_vertical_flip"]))

    if "random_rotation" in transform_config:
        aug_list.append(A.Rotate(**transform_config["random_rotation"]))

    if "perspective" in transform_config:
        aug_list.append(A.Perspective(**transform_config["perspective"]))

    # (4) Pixel-Level (Color, Sharpness, etc.)
    if "color_jitter" in transform_config:
        aug_list.append(A.ColorJitter(**transform_config["color_jitter"]))

    if "random_grayscale" in transform_config:
        aug_list.append(A.ToGray(**transform_config["random_grayscale"]))

    if "sharpen" in transform_config:
        aug_list.append(A.Sharpen(**transform_config["sharpen"]))

    if "clahe" in transform_config:
        aug_list.append(A.CLAHE(**transform_config["clahe"]))

    if "gaussian_noise" in transform_config:
        aug_list.append(A.GaussNoise(**transform_config["gaussian_noise"]))

    if "gaussian_blur" in transform_config:
        aug_list.append(A.GaussianBlur(**transform_config["gaussian_blur"]))

    if "solarize" in transform_config:
        aug_list.append(A.Solarize(**transform_config["solarize"]))

    # (5) Structural / Occlusion
    if "coarse_dropout" in transform_config:
        aug_list.append(A.CoarseDropout(**transform_config["coarse_dropout"]))

    # (6) Normalize
    if "normalize" in transform_config:
        aug_list.append(A.Normalize(**transform_config["normalize"]))

    # Finally, convert to PyTorch Tensor
    aug_list.append(ToTensorV2())

    return A.Compose(aug_list)
