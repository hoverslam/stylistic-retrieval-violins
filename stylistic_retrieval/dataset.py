from __future__ import annotations

from stylistic_retrieval.utils.visuals import Image, load_image
from stylistic_retrieval.utils.io import read_json

import logging
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any

import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset, Sampler


logger = logging.getLogger(__name__)


@dataclass
class Instrument:
    id: str
    maker: str
    place_of_production: str
    date_of_production: str
    images: list[Image]

    def to_dict(self) -> dict[str, str | int]:
        """Convert the instrument metadata to a dictionary.

        Returns:
            dict: Dictionary containing:
                - "id": Instrument identifier.
                - "maker": Instrument maker.
                - "place": Place of production.
                - "date": Date of production.
                - "num_images": Number of associated images.
        """
        return {
            "id": self.id,
            "maker": self.maker,
            "place": self.place_of_production,
            "date": self.date_of_production,
        }

    def get_display_image(self, viewpoint: str = "violin-front") -> tuple[np.ndarray, Image] | None:
        """
        Return an image (or cropped region) representing a requested viewpoint.

        The method searches through the instrument's images for an Area of Interest (AOI)
        whose label matches the requested viewpoint. If found, the corresponding region
        of the image is cropped and returned.

        If no matching AOI is found, the full first image is returned as a fallback.

        Args:
            viewpoint: Label identifying the desired viewpoint (e.g., "violin-front").

        Returns:
            A tuple `(pixels, image)` where:
                - pixels: NumPy array representing the cropped region or full image.
                - image: The `Image` object from which the pixels were taken.

            Returns `None` if the instrument has no images.
        """
        if not self.images:
            return None

        for img_obj in self.images:
            for aoi in img_obj.aois:
                if aoi["label"] == viewpoint:
                    img_np = img_obj.numpy()
                    x_min, y_min, x_max, y_max = aoi["coordinates"]
                    cropped = img_np[y_min:y_max, x_min:x_max, :]
                    pixels = cropped if cropped.size > 0 else img_np
                    return pixels, img_obj

        return self.images[0].numpy(), self.images[0]


class InstrumentDataset(ABC):
    def __init__(self, root: Path | str, meta_file: str, aoi_file: str) -> None:
        super().__init__()
        self.root = Path(root)
        self.instruments = self._load_instruments(meta_file, aoi_file)
        self._id_to_idx = {self.instruments[i].id: i for i in range(len(self.instruments))}

    @property
    def statistics(self) -> dict[str, float | int]:
        return {
            "num_instruments": len(self.instruments),
            "num_makers": len(set([instrument.maker for instrument in self.instruments])),
            "num_images": sum(len(instrument.images) for instrument in self.instruments),
        }

    def get_instrument_by_id(self, id: str) -> Instrument:
        return self.instruments[self._id_to_idx[id]]

    def split(self, split_file: str) -> tuple["InstrumentDataset", "InstrumentDataset", "InstrumentDataset"]:
        """Splits the dataset into three fixed subsets (train, validation, test)
        based on a split file.

        The split file should be a JSON object mapping instrument IDs to integers:
        0 for train, 1 for validation, and 2 for test.

        Args:
            split_file: Path to the JSON file, relative to the meta directory.

        Returns:
            A tuple containing three new InstrumentDataset instances:
            (train_dataset, validation_dataset, test_dataset).
        """
        import copy

        split_data = read_json(self.root / "meta" / split_file)

        # Create deep copies of the dataset instance to serve as templates for the splits.
        # This ensures that each new dataset is an independent object.
        train_dataset = copy.deepcopy(self)
        val_dataset = copy.deepcopy(self)
        test_dataset = copy.deepcopy(self)

        # Prepare empty lists to hold the instruments for each split.
        train_instruments, val_instruments, test_instruments = [], [], []

        # Iterate through the original instruments and distribute them based on the split file.
        for instrument in self.instruments:
            split_idx = split_data.get(instrument.id)
            if split_idx == 0:
                train_instruments.append(instrument)
            elif split_idx == 1:
                val_instruments.append(instrument)
            elif split_idx == 2:
                test_instruments.append(instrument)
            else:
                logger.warning(
                    "Instrument %s missing from split file or invalid split index %s; excluding from all splits.",
                    instrument.id,
                    split_idx,
                )

        # Assign the sorted lists of instruments to the new dataset instances.
        train_dataset.instruments = train_instruments
        val_dataset.instruments = val_instruments
        test_dataset.instruments = test_instruments

        # Re-build the internal 'id to index' mapping for each new dataset,
        # as the indices have changed within the smaller lists.
        for ds in (train_dataset, val_dataset, test_dataset):
            ds._id_to_idx = {instrument.id: i for i, instrument in enumerate(ds.instruments)}

        return train_dataset, val_dataset, test_dataset

    @abstractmethod
    def _load_instruments(self, meta_file: str, aoi_file: str) -> list[Instrument]:
        pass

    def __len__(self) -> int:
        return len(self.instruments)

    def __getitem__(self, idx: int) -> Instrument:
        return self.instruments[idx]


class PaschRichtersDataset(InstrumentDataset):
    def __init__(
        self,
        root: Path | str,
        meta_file: str,
        aoi_file: str = "",
    ) -> None:
        super().__init__(root, meta_file, aoi_file)

    def _load_instruments(self, meta_file: str, aoi_file: str) -> list[Instrument]:
        meta_data = read_json(self.root / "meta" / meta_file)
        aoi_data = read_json(self.root / "meta" / aoi_file) if aoi_file else None

        instruments = []
        for data in meta_data:
            images = []
            for img_path in data["images"]:
                images.append(
                    Image(
                        path=(self.root / img_path).resolve(strict=True),
                        aois=aoi_data.get(img_path) if aoi_data else [],
                    )
                )
            instruments.append(
                Instrument(
                    id=data["id"],
                    maker=data["maker"],
                    place_of_production=data["place_of_production"],
                    date_of_production=data["date_of_production"],
                    images=images,
                )
            )

        return instruments


class MIMODataset(InstrumentDataset):
    def __init__(
        self,
        root: Path | str,
        meta_file: str,
        aoi_file: str = "",
    ) -> None:
        super().__init__(root, meta_file, aoi_file)

    def _load_instruments(self, meta_file: str, aoi_file: str) -> list[Instrument]:
        meta_data = read_json(self.root / "meta" / meta_file)
        aoi_data = read_json(self.root / "meta" / aoi_file) if aoi_file else None

        instruments = []
        for data in meta_data:
            if data["maker"] == "N/A":
                continue

            images = []
            for img_path in data["images"]:
                images.append(
                    Image(
                        path=(self.root / img_path).resolve(strict=True),
                        aois=aoi_data.get(img_path) if aoi_data else [],
                    )
                )
            instruments.append(
                Instrument(
                    id=data["id"],
                    maker=data["maker"],
                    place_of_production=data["place_of_production"],
                    date_of_production=data["date_of_production"],
                    images=images,
                )
            )

        return instruments


class TorchDatasetWrapper(Dataset):
    def __init__(
        self,
        dataset: InstrumentDataset,
        filters: dict[str, list[str]] | None = None,
        transform: A.Compose | None = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.filters = filters if filters else {}
        self.transform = transform
        self._samples, self._labels = self._build_dataset(self.filters)
        self._label_to_idx = {label: i for i, label in enumerate(set(self._labels))}

    @property
    def labels(self) -> list[int]:
        return [self._label_to_idx[label] for label in self._labels]

    @property
    def classes(self) -> list[str]:
        return sorted(list(set(self._labels)))

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def _build_dataset(self, filters: dict[str, list[str]]) -> tuple[list[tuple[int, int]], list[str]]:
        aoi_filters = filters.get("aois")

        samples, labels = [], []
        for instrument_idx, instrument in enumerate(self.dataset.instruments):
            for image_idx, image in enumerate(instrument.images):
                if not aoi_filters:
                    # If there are no AOI filters we store the image => (instrument_idx, image_idx)
                    samples.append((instrument_idx, image_idx))
                    labels.append(instrument.maker)
                else:
                    # Otherwise, we store the AOI => (instrument_idx, image_idx, aoi_idx)
                    if image.aois:
                        for aoi_idx, aoi in enumerate(image.aois):
                            if aoi["label"] not in aoi_filters:
                                continue

                            samples.append((instrument_idx, image_idx, aoi_idx))
                            labels.append(instrument.maker)

        return samples, labels

    def get_instrument(self, idx: int) -> Instrument:
        instrument_idx = self._samples[idx][0]

        return self.dataset.instruments[instrument_idx]

    def get_image(self, idx: int) -> Image:
        instrument_idx = self._samples[idx][0]
        image_idx = self._samples[idx][1]

        return self.dataset.instruments[instrument_idx].images[image_idx]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self._samples[idx]
        instrument_idx = sample[0]
        image_idx = sample[1]
        image_obj = self.dataset[instrument_idx].images[image_idx]
        img_numpy = image_obj.numpy()

        if len(sample) == 3:
            # This means the sample is from an AOI
            aoi_idx = sample[2]
            x_min, y_min, x_max, y_max = image_obj.aois[aoi_idx]["coordinates"]

            # Attempt to slice with original coordinates
            cropped_img = img_numpy[y_min:y_max, x_min:x_max, :]

            # Check if the result is empty (size 0) or invalid
            if cropped_img.size == 0:
                logger.error(
                    "Invalid AOI for image %s: coordinates %s resulted in empty slice. Falling back to full image.",
                    image_obj.path,
                    (x_min, y_min, x_max, y_max),
                )
            else:
                img_numpy = cropped_img

        if self.transform:
            img = self.transform(image=img_numpy)["image"]
        else:
            img = torch.from_numpy(img_numpy)

        target = self._label_to_idx[self._labels[idx]]
        target = torch.tensor(target, dtype=torch.long)

        return img, target


class InstrumentBatchSampler(Sampler[list[int]]):
    """Yield one batch per instrument, i.e. each run of consecutive indices sharing an instrument ID.

    Needed for models whose embeddings depend on the other images in the batch (e.g. CricaVPR's
    cross-image encoder), so that no information leaks between instruments. Index order is preserved.
    """

    def __init__(self, dataset: TorchDatasetWrapper) -> None:
        self._batches: list[list[int]] = []
        prev_id = None
        for idx in range(len(dataset)):
            instrument_id = dataset.get_instrument(idx).id
            if instrument_id != prev_id:
                self._batches.append([])
                prev_id = instrument_id
            self._batches[-1].append(idx)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


class SurveyDataset(Dataset):
    def __init__(self, root: Path | str, meta_file: str, transform: A.Compose | None = None) -> None:
        super().__init__()
        self.root = Path(root)
        self.meta_file = meta_file
        self.transform = transform
        self._instruments = self._built_dataset()

    def _built_dataset(self) -> list[dict[str, Any]]:
        meta_data = read_json(self.root / self.meta_file)
        instruments = []
        for i, question_group in enumerate(meta_data):
            group_id = i + 1
            # Append reference
            instruments.append({"group": group_id, "type": "R", "meta": question_group["reference"]})

            # Append candidates
            for k, v in question_group["candidates"].items():
                instruments.append({"group": group_id, "type": k, "meta": v})

        return instruments

    def get_instrument(self, idx: int) -> dict[str, Any]:
        return self._instruments[idx]

    def __len__(self) -> int:
        return len(self._instruments)

    def __getitem__(self, idx: int) -> torch.Tensor:
        instrument = self._instruments[idx]
        # Create images file name
        if instrument["type"] == "R":
            img_file = f"{instrument['group']}_reference.jpg"
        else:
            img_file = f"{instrument['group']}_candidate{instrument['type']}.jpg"

        img_path = self.root / "images" / img_file
        img_numpy = load_image(img_path, color_mode="RGB")

        if self.transform:
            img = self.transform(image=img_numpy)["image"]
        else:
            img = torch.from_numpy(img_numpy)

        return img
