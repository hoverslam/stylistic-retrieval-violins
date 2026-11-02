from stylistic_retrieval.dataset import InstrumentBatchSampler, InstrumentDataset, Instrument, TorchDatasetWrapper
from stylistic_retrieval.models.retrieval import setup_retrieval_model
from stylistic_retrieval.utils.preprocessing import setup_transform
from stylistic_retrieval.utils.config import Config

import json
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image as PILImage

import matplotlib.pyplot as plt


@dataclass
class Gallery:
    items: dict[str, dict[str, Any]] = field(default_factory=dict)
    embeddings: list[np.ndarray] = field(default_factory=list)
    idx_to_instrument_id: dict[int, str] = field(default_factory=dict)

    def add_item(self, instrument: Instrument, image: torch.Tensor, embedding: np.ndarray) -> None:
        if instrument.id not in self.items:
            self.items[instrument.id] = {"instrument": instrument, "images": [image], "embeddings": [embedding]}
        else:
            self.items[instrument.id]["images"].append(image)
            self.items[instrument.id]["embeddings"].append(embedding)

    def aggregate_embeddings(self) -> None:
        for idx, (instrument_id, instrument) in enumerate(self.items.items()):
            stacked = np.stack(instrument["embeddings"], axis=0)
            mean_embedding = np.mean(stacked, axis=0)
            mean_embedding /= np.linalg.norm(mean_embedding) + 1e-10

            self.embeddings.append(mean_embedding)
            self.idx_to_instrument_id[idx] = instrument_id


class Retriever:
    """
    A class to handle embedding-based retrieval of instruments.

    This class encapsulates the process of loading a pre-trained model,
    extracting feature vectors (embeddings) from a dataset to create a searchable gallery,
    and performing retrieval for new query instruments.
    """

    def __init__(self, dataset: InstrumentDataset, model_path: Path | str, config: Config, device: str) -> None:
        """
        Initializes the Retriever.

        Args:
            dataset (TorchDatasetWrapper): The dataset to use as the search gallery.
            model_dir (Path | str): Path to the directory containing the pre-trained model file.
            config (Config): The configuration object used for model setup.
            device (str): The device to run the model on ('cuda' or 'cpu').
        """
        self.model_path = Path(model_path)
        self.config = config
        self.device = torch.device(device)
        self._model = self._load_model(self.model_path)
        self._transform = setup_transform(self.config, mode="test")
        self._built_gallery(dataset)

    def _load_model(self, model_path: Path) -> torch.nn.Module:
        """Loads the retrieval model from a file."""

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at '{model_path.as_posix()}'")

        model = setup_retrieval_model(self.config)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        return model

    def _built_gallery(self, dataset: InstrumentDataset) -> None:
        # Load data
        data = TorchDatasetWrapper(dataset, self.config["dataset"].get("filters", {}), transform=self._transform)
        if self.config.as_dict.get("test_settings", {}).get("batch_by_instrument", False):
            # One batch per instrument (in dataset order) for models with batch-dependent embeddings
            loader = DataLoader(data, batch_sampler=InstrumentBatchSampler(data), num_workers=8, pin_memory=True)
        else:
            loader = DataLoader(
                data,
                batch_size=self.config["train_settings"]["batch_size"],
                num_workers=8,
                pin_memory=True,
                shuffle=False,
            )

        # Extract image embeddings
        all_embeddings = []
        with torch.no_grad():
            for images, _ in tqdm(loader, desc="Indexing Gallery"):
                images = images.to(self.device)
                embeddings = self._model(images).cpu()
                all_embeddings.append(embeddings)
        all_embeddings = torch.cat(all_embeddings).numpy()

        # Built gallery that stores the instrument embeddings
        self.gallery = Gallery()
        for i, embedding in enumerate(all_embeddings):
            self.gallery.add_item(instrument=data.get_instrument(i), image=data[i][0], embedding=embedding)
        self.gallery.aggregate_embeddings()

    def retrieve(self, query_instruments: list[Instrument], top_k: int) -> list[list[dict[str, Any]]]:
        """Retrieve the most similar instruments using aggregated query embeddings.

        This method mirrors the gallery construction by extracting features for all
        images associated with a query instrument that match the specified AOIs,
        averaging them into a single representative vector, and performing
        a cosine similarity search.

        Args:
            query_instruments: List of instrument objects to search for.
            top_k: Number of unique similar instruments to return per query.

        Returns:
            A list of retrieval results for each query, where each result is a
            list of dictionaries containing the retrieved instrument,
            similarity score, and its associated gallery images.

        Raises:
            ValueError: If no valid images are found for a query instrument
                given the specified AOI labels.
        """

        aoi_labels = self.config["dataset"].get("filters", {}).get("aois", [])
        query_embeddings = []
        for instrument in query_instruments:
            instrument_images = []

            # Identify images containing at least one of the target AOI labels
            for img_obj in instrument.images:
                has_aoi = any(aoi["label"] in aoi_labels for aoi in img_obj.aois)
                if has_aoi:
                    instrument_images.append(img_obj.numpy())

            if not instrument_images:
                raise ValueError(f"Instrument {instrument.id} has no images matching {aoi_labels}")

            # Batch process all relevant images for the current instrument
            processed = [self._transform(image=img)["image"] for img in instrument_images]
            batch = torch.stack(processed).to(self.device)

            with torch.no_grad():
                features = self._model(batch).cpu().numpy()

            # Calculate the centroid of the image features and normalize to the unit hypersphere
            mean_emb = np.mean(features, axis=0)
            mean_emb /= np.linalg.norm(mean_emb) + 1e-10
            query_embeddings.append(mean_emb)

        # Matrix multiplication performs efficient cosine similarity on normalized vectors
        query_features = np.stack(query_embeddings, axis=0)
        gallery_embeddings = np.stack(self.gallery.embeddings, axis=0)
        sims_matrix = query_features @ gallery_embeddings.T

        all_results = []
        for i, query_instrument in enumerate(query_instruments):
            query_results = []
            seen_ids = {query_instrument.id}

            # Retrieve more than top_k to account for the query instrument's own ID being filtered out
            top_indices = np.argsort(-sims_matrix[i])[: top_k + 10]

            for idx in top_indices:
                instr_id = self.gallery.idx_to_instrument_id[idx]
                if instr_id not in seen_ids:
                    item = self.gallery.items[instr_id]
                    query_results.append(
                        {
                            "instrument": item["instrument"],
                            "score": float(sims_matrix[i, idx]),
                            "images": item["images"],
                        }
                    )
                    seen_ids.add(instr_id)

                if len(query_results) >= top_k:
                    break

            all_results.append(query_results)

        return all_results


def plot_retrieval_results(query_instruments: list[Instrument], results: list[list[dict[str, Any]]]) -> None:
    """
    Plots a grid of retrieval results.

    Each row corresponds to a query. The first column shows the query image
    (blue border), and subsequent columns show the retrieved results.
    A green border indicates a correct maker match, while red indicates a mismatch.

    Args:
        query_instruments (list[Instrument]): The list of query instruments.
        results (list[list[dict]]): The retrieval results from the Retriever class.
    """
    if not query_instruments:
        print("No queries to plot.")
        return

    num_queries = len(query_instruments)
    # Determine number of columns from the first result entry, +1 for the query
    num_results = len(results[0]) if results else 0
    num_cols = 1 + num_results

    _, axes = plt.subplots(
        num_queries,
        num_cols,
        figsize=(num_cols * 3, num_queries * 3.5),
        squeeze=False,  # Always return a 2D array for axes
    )

    for row, (query_instrument, query_results) in enumerate(zip(query_instruments, results)):
        # Display the Query Image (First Column)
        ax = axes[row, 0]
        result_data = query_instrument.get_display_image(viewpoint="top-plate")
        if result_data:
            query_pixels, query_img_obj = result_data
            ax.imshow(query_pixels)
            title = f"{query_instrument.maker}\n{query_img_obj.name}"
            ax.set_title(title, fontsize=8, color="blue")

            # Set blue border for query
            for spine in ax.spines.values():
                spine.set_edgecolor("blue")
                spine.set_linewidth(3)
        ax.set_xticks([])
        ax.set_yticks([])

        # Display the Result Images
        for col, result in enumerate(query_results, start=1):
            ax = axes[row, col]
            retrieved_instrument = result["instrument"]
            result_data = retrieved_instrument.get_display_image(viewpoint="top-plate")

            if result_data:
                # Determine border color
                is_correct = retrieved_instrument.maker == query_instrument.maker
                border_color = "green" if is_correct else "red"

                res_pixels, res_img_obj = result_data
                ax.imshow(res_pixels)
                title = f"{retrieved_instrument.maker}\n{res_img_obj.name}"
                ax.set_title(title, fontsize=8)

                # Set border color based on correctness
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3)
            ax.set_xticks([])
            ax.set_yticks([])

    # Hide any unused axes
    for i in range(num_queries):
        for j in range(num_cols):
            if not axes[i, j].images:
                axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()


def save_retrieval_results(
    query_instruments: list[Instrument],
    results: list[list[dict[str, Any]]],
    output_dir: str | Path,
) -> None:
    """Save retrieval results as images and JSON files.

    For each query instrument:
      - A folder is created under ``output_dir`` named after the instrument id.
      - The query image is saved as ``[0] instrument_id.jpg``.
      - Retrieved results are saved as ``[i] instrument_id.jpg``.
      - A ``results.json`` file stores structured metadata.

    Args:
        query_instruments: List of query instruments.
        results: Output from the retriever.
        output_dir: Directory where results will be stored.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for query_instrument, query_results in zip(query_instruments, results):
        instrument_dir = output_dir / str(query_instrument.id)
        instrument_dir.mkdir(parents=True, exist_ok=True)

        json_data: dict[str, Any] = {"query": query_instrument.to_dict(), "results": []}

        # Save query image
        query_img = query_instrument.get_display_image(viewpoint="top-plate")
        if query_img:
            pixels, _ = query_img
            filename = f"[0] {query_instrument.id}.jpg"
            PILImage.fromarray(pixels).save(instrument_dir / filename)

        for i, result in enumerate(query_results, start=1):
            retrieved = result["instrument"]

            # Save retrieved image
            result_img = retrieved.get_display_image(viewpoint="top-plate")
            if result_img:
                pixels, _ = result_img
                filename = f"[{i}] {retrieved.id}.jpg"
                PILImage.fromarray(pixels).save(instrument_dir / filename)

            # Add query results to final dictionary
            entry = retrieved.to_dict()
            entry["rank"] = i
            entry["score"] = result["score"]

            json_data["results"].append(entry)

        # Save JSON
        with open(instrument_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
