from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

from stylistic_retrieval.dataset import InstrumentBatchSampler, TorchDatasetWrapper
from stylistic_retrieval.utils.config import Config

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """
    Performs feature extraction and retrieval-based evaluation for a model–dataset
    pair. Supports image-level and instrument-level (aggregated) embeddings and
    computes standard retrieval metrics including mAP, Top-1 accuracy, and a suite
    of @K statistics (Hard@K, Soft@K, Precision@K).
    """

    def __init__(self, model: torch.nn.Module, config: Config, device: str, num_workers: int) -> None:
        """
        Initialize the evaluator.

        Args:
            model (torch.nn.Module): The model to evaluate.
            config (Config): Configuration object with test settings.
            device (str): Device for running inference ("cuda" or "cpu").
            num_workers (int): Number of worker processes for data loading.
        """
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.num_workers = num_workers

        self.pca = None

        # Extract settings from the config dictionary
        test_settings_cfg = self.config.as_dict.get("test_settings", {})
        self.batch_size = test_settings_cfg.get("batch_size", 32)
        self.distance_metric = test_settings_cfg.get("distance", "euclidean")
        self.aggregation = test_settings_cfg.get("aggregation")
        self.max_k = test_settings_cfg.get("max_k", 10)
        self.pca_params = test_settings_cfg.get("pca_params", test_settings_cfg.get("pca"))
        self.batch_by_instrument = test_settings_cfg.get("batch_by_instrument", False)

        # Define metric calculation logic once to ensure consistency across
        # the main evaluation and the empty/fallback states.
        self._k_metric_defs: dict[str, Callable[[np.ndarray, int], float]] = {
            "hard": lambda mask, k: float(np.all(mask[:k])),
            "soft": lambda mask, k: float(np.any(mask[:k])),
            "precision": lambda mask, k: float(np.mean(mask[:k])),
        }

    def evaluate(self, dataset: TorchDatasetWrapper, prefix: str = "") -> dict[str, Any]:
        """
        Run full evaluation: extract features, aggregate (optional), and compute metrics.

        Args:
            dataset (TorchDatasetWrapper): Dataset to evaluate.
            prefix (str, optional): String added to the metric keys. Defaults to "".

        Returns:
            dict[str, Any]: Computed retrieval metrics including summary means and a 'raw'
                dictionary for per-query analysis.
        """
        features, labels, ids = self._extract_features(dataset, desc=f"Evaluating {prefix.strip('_')}")

        # Reduce multiple views/images of the same instrument into a single
        # representative vector if specified in the config.
        features, labels, ids = self._aggregate_features(features, labels, ids)

        if self.pca_params:
            features = self._reduce_dimensionality(features)

        raw_metrics = self._compute_metrics(features, labels, ids)

        return {f"{prefix}{k}": v for k, v in raw_metrics.items()}

    def fit_pca(self, dataset: TorchDatasetWrapper) -> None:
        """
        Fits the PCA transformation on a reference dataset (usually training)
        to capture feature distributions without leaking evaluation data.

        Args:
            dataset (TorchDatasetWrapper): Dataset used to compute PCA components.
        """
        if not self.pca_params:
            return

        features, labels, ids = self._extract_features(dataset, desc="Fitting PCA")
        features, _, _ = self._aggregate_features(features, labels, ids)

        self.pca = PCA(**self.pca_params)
        self.pca.fit(features)

    def _extract_features(self, dataset: TorchDatasetWrapper, desc: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract embeddings, labels, and instrument IDs using the model.

        Args:
            dataset (TorchDatasetWrapper): Dataset to process without shuffling.
            desc (str): Description shown in progress display.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of features, labels, and instrument IDs.
        """
        if self.batch_by_instrument:
            # One batch per instrument (in dataset order) for models with batch-dependent embeddings
            loader = DataLoader(
                dataset,
                batch_sampler=InstrumentBatchSampler(dataset),
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=False,
            )

        all_embeddings = []
        all_labels = []

        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            for images, targets in tqdm(loader, desc=desc):
                images = images.to(self.device)
                embeddings = self.model(images)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(targets)

        if len(all_embeddings) > 0:
            features = torch.cat(all_embeddings).numpy()
            labels = torch.cat(all_labels).numpy()
        else:
            features = np.array([])
            labels = np.array([])

        # We rely on shuffle=False to ensure that the sequential iteration of
        # the dataset matches the order of features coming out of the DataLoader.
        ids = np.array([dataset.get_instrument(i).id for i in range(len(dataset))])

        return features, labels, ids

    def _aggregate_features(
        self, features: np.ndarray, labels: np.ndarray, ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aggregate image-level embeddings into instrument-level embeddings.

        Args:
            features (np.ndarray): Feature vectors of shape (N, D).
            labels (np.ndarray): Labels for each feature vector.
            ids (np.ndarray): Instrument IDs aligned with features.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Aggregated features with their labels and ids.
        """
        # If there is no aggregation we add a suffix to the instrument ID to make it unqiue
        if self.aggregation is None:
            id_counts = {}
            unique_ids = []
            for uid in ids:
                uid_str = str(uid)
                count = id_counts.get(uid_str, 0)
                unique_ids.append(f"{uid_str}_{count}")
                id_counts[uid_str] = count + 1

            return features, labels, np.array(unique_ids)

        # If we aggregate the features the IDs are unqiue
        if self.aggregation.lower() == "mean":
            unique_ids = np.unique(ids)
            agg_features = []
            agg_labels = []

            for uid in unique_ids:
                indices = np.where(ids == uid)[0]
                inst_features = features[indices]
                mean_feature = np.mean(inst_features, axis=0)

                # Stability check & Re-normalization
                if not np.all(np.isfinite(mean_feature)) or np.linalg.norm(mean_feature) < 1e-12:
                    mean_feature = np.random.normal(0, 1e-6, size=mean_feature.shape)

                mean_feature = np.asarray(normalize(mean_feature.reshape(1, -1), norm="l2")).flatten()

                agg_features.append(mean_feature)
                agg_labels.append(labels[indices[0]])

            return np.array(agg_features), np.array(agg_labels), unique_ids

        raise ValueError(f"Unknown aggregation method: {self.aggregation}")

    def _reduce_dimensionality(self, features: np.ndarray) -> np.ndarray:
        """
        Apply pre-fitted PCA and re-normalize to maintain metric consistency.

        Args:
            features (np.ndarray): High-dimensional feature matrix.

        Returns:
            np.ndarray: L2-normalized, whitened, reduced-dimension features.
        """
        if self.pca is None:
            # Fitting on test data is avoided to prevent data leakage
            # and artificially inflated retrieval scores.
            raise RuntimeError("PCA must be fitted via fit_pca() before calling evaluate().")

        features = self.pca.transform(features)

        # We re-normalize here because whitening and rotation can move
        # points off the unit hypersphere used by our distance metrics.
        features = np.asarray(normalize(features, norm="l2"))

        return features

    def _compute_metrics(self, features: np.ndarray, labels: np.ndarray, ids: np.ndarray) -> dict[str, Any]:
        """
        Compute summary and per-query metrics for the retrieval set.

        Args:
            features (np.ndarray): Embedding matrix of shape (N, D).
            labels (np.ndarray): Ground-truth labels of shape (N,).

        Returns:
            dict[str, Any]: Dictionary containing mean values and a 'raw' sub-dict for
                statistical testing (Wilcoxon/T-test).
        """
        eval_range = range(1, self.max_k + 1)

        if len(labels) == 0:
            return self._get_empty_results()

        # Queries with only one instance in the dataset (singletons) are
        # excluded as they have no valid targets to retrieve in the gallery.
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique_labels, counts))
        num_samples = len(labels)

        # Batch compute all distances to find neighbors for every sample.
        dists = pairwise_distances(features, metric=self.distance_metric)

        # Initialize storage for every query to enable post-hoc significance testing.
        raw_results: dict[str, list] = {"id": [], "label": [], "ap": [], "top-1": []}
        for k in eval_range:
            for name in self._k_metric_defs:
                raw_results[f"{name}@{k}"] = []

        valid_sample_count = 0
        for i in range(num_samples):
            true_label = labels[i]
            if label_counts[true_label] < 2:
                continue

            valid_sample_count += 1
            sorted_idxs = np.argsort(dists[i])

            # The query image itself is always at index 0 (distance=0); we
            # remove it to evaluate the model's ability to find *other* matches.
            sorted_labels = labels[sorted_idxs][1:]

            if len(sorted_labels) == 0:
                continue

            raw_results["id"].append(ids[i])
            raw_results["label"].append(true_label)

            # Top-1 is isolated for display purposes as requested,
            # though it is mathematically identical to Precision@1.
            is_top1 = 1.0 if sorted_labels[0] == true_label else 0.0
            raw_results["top-1"].append(is_top1)

            # mAP rewards models for placing all relevant items at the top
            # of the ranked list.
            relevant_ranks = np.where(sorted_labels == true_label)[0]
            if len(relevant_ranks) > 0:
                relevant_ranks_1b = relevant_ranks + 1
                precisions = np.arange(1, len(relevant_ranks) + 1) / relevant_ranks_1b
                raw_results["ap"].append(float(np.mean(precisions)))
            else:
                raw_results["ap"].append(0.0)

            # Iteratively apply metric functions to the neighbor mask.
            is_correct = sorted_labels == true_label
            for k in eval_range:
                limit = min(k, len(is_correct))
                for name, func in self._k_metric_defs.items():
                    raw_results[f"{name}@{k}"].append(func(is_correct, limit))

        if valid_sample_count == 0:
            return self._get_empty_results()

        # Construct final output by averaging the lists of per-query scores.
        metrics: dict[str, Any] = {
            "map": np.mean(raw_results["ap"]),
            "top-1": np.mean(raw_results["top-1"]),
        }
        for key, values in raw_results.items():
            if "@" in key:
                metrics[key] = np.mean(values)

        metrics["raw"] = raw_results
        return metrics

    def _get_empty_results(self) -> dict[str, Any]:
        """
        Generates a consistent zeroed-out results dictionary for edge cases.

        Returns:
            dict[str, Any]: Initialized results matching the structure of a valid run.
        """
        out: dict[str, Any] = {"map": 0.0, "top-1": 0.0, "raw": {}}
        for k in range(1, self.max_k + 1):
            for name in self._k_metric_defs:
                out[f"{name}@{k}"] = 0.0
        return out
