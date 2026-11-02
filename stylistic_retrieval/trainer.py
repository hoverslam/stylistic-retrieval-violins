from stylistic_retrieval.utils.config import Config
from stylistic_retrieval.dataset import TorchDatasetWrapper
from stylistic_retrieval.evaluator import RetrievalEvaluator

import sys
import random
import shutil
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR
from pytorch_metric_learning.losses import (
    TripletMarginLoss,
    ContrastiveLoss,
    ProxyAnchorLoss,
    ArcFaceLoss,
    MultiSimilarityLoss,
    SupConLoss,
)
from pytorch_metric_learning.distances import LpDistance, CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.miners import (
    TripletMarginMiner,
    PairMarginMiner,
    MultiSimilarityMiner,
    DistanceWeightedMiner,
)
from pytorch_metric_learning.samplers import MPerClassSampler
import wandb

StopperMetric = namedtuple("StopperMetric", ["name", "value"])


def set_seed(seed: int = 42) -> None:
    """
    Set all relevant random seeds for reproducible training.

    Args:
        seed (int, optional): Seed value to use. Defaults to 42.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """
    Utility for stopping training early when a monitored metric stops improving.

    Args:
        patience (int): Number of epochs to wait before stopping.
        warmup (int): Number of initial epochs to ignore for early stopping.
        maximize (bool): Whether a higher metric value indicates improvement.
        verbose (bool, optional): Whether to print status messages. Defaults to True.
    """

    def __init__(self, patience: int, warmup: int, maximize: bool, verbose: bool = True) -> None:
        self.patience = patience
        self.warmup = warmup
        self.maximize = maximize
        self.verbose = verbose
        self.counter = 0
        self.best_metric = float("-inf") if maximize else float("inf")

    def __call__(self, metric: StopperMetric, epoch: int) -> bool:
        """
        Evaluate whether early stopping should occur.

        Args:
            metric (StopperMetric): Metric object containing name and value.
            epoch (int): Current epoch number.

        Returns:
            bool: True if training should stop, otherwise False.
        """
        if epoch < self.warmup:
            return False
        is_better = (metric.value > self.best_metric) if self.maximize else (metric.value < self.best_metric)
        if is_better:
            self.best_metric, self.counter = metric.value, 0
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"⏳ Early stopping: {self.counter}/{self.patience}. Best metric: {metric.name}={self.best_metric:.6f}"
                )
            if self.counter >= self.patience:
                if self.verbose:
                    print("🛑 Early stopping triggered.")
                return True
        return False


class RetrievalTrainer:
    """
    A self-contained trainer for embedding-based retrieval models.

    Args:
        model: Model to train.
        train_data (TorchDatasetWrapper): Training dataset.
        eval_data (dict[str, TorchDatasetWrapper]): Dict of evaluation datasets (e.g., {"train": ..., "val": ...}).
        config (Config): Training and evaluation configuration.
        device (str): Device identifier ("cuda" or "cpu").
        num_workers (int): Number of worker processes for data loading.
    """

    def __init__(
        self,
        model,
        train_data: TorchDatasetWrapper,
        eval_data: dict[str, TorchDatasetWrapper],
        config: Config,
        device: str,
        num_workers: int,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_data = train_data
        self.eval_data = eval_data
        self.config = config
        self.num_workers = num_workers
        self.evaluator = RetrievalEvaluator(self.model, self.config, device, self.num_workers)

    def run(self) -> None:
        """Run the complete training loop, including evaluation, logging, and early stopping."""
        set_seed(42)
        train_settings = self.config["train_settings"]
        stopper = EarlyStopping(**train_settings["early_stopping"]) if "early_stopping" in train_settings else None

        # Setup training components
        distance = self._setup_distance()
        reducer = self._setup_reducer()
        criterion = self._setup_criterion(distance, reducer)
        optimizer = self._setup_optimizer(criterion)
        scheduler = self._setup_scheduler(optimizer)
        miner = self._setup_miner(distance)
        sampler = self._setup_sampler()

        # Load training data
        train_loader = DataLoader(
            self.train_data,
            batch_size=train_settings["batch_size"],
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=self.num_workers,
            pin_memory=True,
        )

        with self._init_wandb() as run:
            # Uplaod config YAML to W&B
            config_path = Path(run.dir) / self.config.file_path.name
            shutil.copy(self.config.file_path, config_path)
            wandb.save(str(config_path), base_path=run.dir)

            model_dir = Path(run.dir).parent / "checkpoints"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_name = self.config["model"]["name"]
            model_name = f"{model_name}__{run.id}"

            print("#" * 50 + "\n# TRAINING\n" + "#" * 50)
            print(f"▸ Train samples: {len(self.train_data)} | Val samples: {len(self.eval_data['val'])}")
            print(f"▸ Model: '{model_name}' → '{self.device}'" + "\n")

            # Initial evaluation
            print(f"{'# Initial Evaluation ':-<50}")
            self.evaluator.fit_pca(self.eval_data["train"])
            train_metrics = self.evaluator.evaluate(self.eval_data["train"], prefix="train_")
            val_metrics = self.evaluator.evaluate(self.eval_data["val"], prefix="val_")
            print(
                f"📊 Train [mAP: {train_metrics['train_map']:.4f} | Top-1: {train_metrics['train_top-1']:.4f}]  ||  "
                f"Val [mAP: {val_metrics['val_map']:.4f} | Top-1: {val_metrics['val_top-1']:.4f}]\n"
            )
            run.log({"step": 0, **train_metrics, **val_metrics})

            best_metrics = {"map": -float("inf")}
            best_model_state = {k: v.cpu() for k, v in self._get_raw_state_dict(self.model).items()}

            # Training loop
            print(f"{'# Training ':-<50}")
            global_step, eval_count = 0, 0
            stop_training = False
            max_steps = train_settings["max_steps"]
            with tqdm(total=max_steps, desc="Training", dynamic_ncols=True) as pbar:
                while not stop_training:
                    self.model.train()
                    epoch_loss = 0.0
                    batch_count = 0

                    for images, targets in train_loader:
                        # Check global step limit
                        if global_step >= max_steps:
                            stop_training = True
                            break

                        images, targets = images.to(self.device), targets.to(self.device)
                        embeddings = self.model(images)

                        indices_tuple = miner(embeddings, targets) if miner is not None else None
                        loss = (
                            criterion(embeddings, targets, indices_tuple)
                            if indices_tuple is not None
                            else criterion(embeddings, targets)
                        )

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()

                        # Update metrics
                        epoch_loss += loss.item()
                        batch_count += 1
                        global_step += 1

                        # Update the progress bar and postfix
                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "loss": f"{loss.item():.4f}",
                                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                            }
                        )

                        # Periodic Logging to W&B
                        run.log(
                            {
                                "batch_loss": loss.item(),
                                "step": global_step,
                                "learning_rate": optimizer.param_groups[0]["lr"],
                                "active_triplets": len(indices_tuple[0]) if indices_tuple is not None else 0,
                            }
                        )

                        # Periodic Evaluation
                        if global_step % train_settings.get("eval_every_steps", 500) == 0:
                            avg_loss = epoch_loss / batch_count
                            epoch_loss, batch_count = 0.0, 0

                            self.evaluator.fit_pca(self.eval_data["train"])
                            train_metrics = self.evaluator.evaluate(self.eval_data["train"], prefix="train_")
                            val_metrics = self.evaluator.evaluate(self.eval_data["val"], prefix="val_")

                            # Use tqdm.write to avoid breaking the progress bar line
                            tqdm.write(
                                f"\nStep {global_step}/{max_steps} | Avg. Train Loss: {avg_loss:.6f}\n"
                                f"📊 Train [mAP: {train_metrics['train_map']:.4f} | Top-1: {train_metrics['train_top-1']:.4f}]  ||  "
                                f"Val [mAP: {val_metrics['val_map']:.4f} | Top-1: {val_metrics['val_top-1']:.4f}]"
                            )

                            run.log(
                                {
                                    "step": global_step,
                                    **train_metrics,
                                    **val_metrics,
                                }
                            )

                            # Save best model
                            if val_metrics["val_map"] > best_metrics["map"]:
                                best_model_state = {k: v.cpu() for k, v in self._get_raw_state_dict(self.model).items()}
                                best_metrics["map"] = float(val_metrics["val_map"])

                            # Early Stopping logic
                            eval_count += 1
                            stopper_metric = StopperMetric("mAP", val_metrics["val_map"])
                            if stopper and stopper(stopper_metric, eval_count):
                                stop_training = True
                                break  # Exit the 'for' loop

            # Save best model
            best_model_path = (model_dir / f"{model_name}-best").with_suffix(".pt")
            torch.save(best_model_state, best_model_path)

            # Save final model
            final_model_state = {k: v.cpu() for k, v in self._get_raw_state_dict(self.model).items()}
            final_model_path = (model_dir / f"{model_name}-final").with_suffix(".pt")
            torch.save(final_model_state, final_model_path)

            print(f"{'# Final Evaluation ':-<50}")

            # We evaluate the best and final weights on the validation set
            for state, suffix in [(best_model_state, "best"), (final_model_state, "final")]:
                # Load into the underlying module if DataParallel is used
                if isinstance(self.model, torch.nn.DataParallel):
                    self.model.module.load_state_dict(state)
                else:
                    self.model.load_state_dict(state)

                # You can iterate through all splits in eval_data
                for split_name, split_data in self.eval_data.items():
                    results = self.evaluator.evaluate(split_data, prefix=f"{split_name}_")
                    self._save_excel_results(run, model_name, f"{suffix}_{split_name}", results)

            print(f"\nTraining finished! Models saved to '{model_dir.resolve().as_posix()}'.")

    def _init_wandb(self):
        run_dir = Path(sys.prefix).parent / ".temp/"
        run_dir.mkdir(parents=True, exist_ok=True)

        return wandb.init(dir=run_dir, config=self.config.as_dict, **self.config["wandb"])

    def _get_raw_state_dict(self, m: torch.nn.Module) -> dict:
        if isinstance(m, torch.nn.DataParallel):
            return m.module.state_dict()

        return m.state_dict()

    def _save_excel_results(self, run, model_name: str, suffix: str, metrics: dict):
        """Creates a multi-tab Excel file with results and uploads it to W&B.

        This method generates an Excel workbook containing two sheets: a transposed
        summary of aggregated metrics and the raw per-query scores. The resulting
        file is saved locally and then tracked as a Weights & Biases artifact.

        Args:
            run: The current wandb run object used for directory pathing and saving.
            model_name (str): Name of the model being evaluated (used for filename).
            suffix (str): Description of the data split (e.g., 'val', 'test').
            metrics (dict): Dictionary containing both scalar summary metrics and
                a list of raw scores under a key ending in '_raw'.

        Returns:
            None
        """
        # Find the raw data key (e.g., 'val_raw') which contains the lists
        raw_key = next((k for k in metrics.keys() if k.endswith("_raw")), None)
        if raw_key is None:
            return

        # Summary Tab: Aggregated means only
        summary_data = {k: v for k, v in metrics.items() if k != raw_key}

        # Convert to DataFrame and transpose so metrics are rows instead of columns
        df_summary = pd.DataFrame([summary_data]).T
        df_summary.columns = ["Value"]
        df_summary.index.name = "Metric"

        # Individual_Results Tab: Per-query scores (AP, @K, etc.)
        raw_data = metrics[raw_key]
        df_raw = pd.DataFrame(raw_data)

        excel_path = Path(run.dir) / f"{model_name}_{suffix}_results.xlsx"

        with pd.ExcelWriter(excel_path) as writer:
            # index=True is now required for Summary to keep the Metric names
            df_summary.to_excel(writer, sheet_name="Summary", index=True)
            df_raw.to_excel(writer, sheet_name="Individual_Results", index=False)

        # Upload the artifact to W&B
        wandb.save(str(excel_path), base_path=run.dir)

    def _setup_optimizer(self, criterion):
        params = self.config["train_params"]["optimizer"]
        opt_kwargs = params["kwargs"].copy()

        # Extract general LR and weight decay
        base_lr = opt_kwargs.pop("lr", 0.0001)

        # Group 0: The backbone model
        param_groups = [{"params": self.model.parameters(), "lr": base_lr}]

        # Group 1: The Loss Layer. Check if a specific 'loss_lr' is provided in the config
        loss_lr = opt_kwargs.pop("loss_lr", base_lr)
        if len(list(criterion.parameters())) > 0:
            param_groups.append({"params": criterion.parameters(), "lr": loss_lr})

        if params["class"] == "AdamW":
            return AdamW(param_groups, **opt_kwargs)

        raise ValueError(f"Unknown optimizer: {params['class']}")

    def _setup_scheduler(self, optimizer):
        if "scheduler" not in self.config["train_params"]:
            return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

        params = self.config["train_params"]["scheduler"]
        max_steps = self.config["train_settings"]["max_steps"]

        if params["class"] == "StepLR":
            main_scheduler = StepLR(optimizer, **params["kwargs"])
        elif params["class"] == "CosineAnnealingLR":
            main_scheduler = CosineAnnealingLR(optimizer, **params["kwargs"])
        else:
            raise ValueError(f"Unknown scheduler: {params['class']}")

        # Add Linear Warmup
        warmup_steps = int(0.02 * max_steps)  # first 2% of max_steps
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / warmup_steps))

        # Combine warmup with main scheduler
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps]
        )

    def _setup_distance(self):
        if "distance" not in self.config["train_params"]:
            return None

        params = self.config["train_params"]["distance"]
        if params["class"] == "LpDistance":
            return LpDistance(**params["kwargs"])
        if params["class"] == "CosineSimilarity":
            return CosineSimilarity(**params.get("kwargs", {}))

        raise ValueError(f"Unknown distance: {params['class']}")

    def _setup_reducer(self):
        if "reducer" not in self.config["train_params"]:
            return None

        params = self.config["train_params"]["reducer"]
        if params["class"] == "ThresholdReducer":
            return ThresholdReducer(**params["kwargs"])

        raise ValueError(f"Unknown reducer: {params['class']}")

    def _setup_criterion(self, distance=None, reducer=None):
        params = self.config["train_params"]["criterion"]
        loss_class = params["class"]

        if loss_class == "ProxyAnchorLoss":
            return ProxyAnchorLoss(num_classes=self.train_data.num_classes, **params["kwargs"])
        if loss_class == "ArcFaceLoss":
            return ArcFaceLoss(num_classes=self.train_data.num_classes, **params["kwargs"])
        if loss_class == "TripletMarginLoss":
            return TripletMarginLoss(distance=distance, reducer=reducer, **params["kwargs"])
        if loss_class == "ContrastiveLoss":
            return ContrastiveLoss(distance=distance, reducer=reducer, **params["kwargs"])
        if loss_class == "MultiSimilarityLoss":
            return MultiSimilarityLoss(distance=distance, reducer=reducer, **params["kwargs"])
        if loss_class == "SupConLoss":
            return SupConLoss(**params["kwargs"])

        raise ValueError(f"Unknown criterion: {loss_class}")

    def _setup_miner(self, distance):
        if "miner" not in self.config["train_params"]:
            return None

        params = self.config["train_params"]["miner"]
        if params["class"] == "TripletMarginMiner":
            return TripletMarginMiner(distance=distance, **params["kwargs"])
        if params["class"] == "PairMarginMiner":
            return PairMarginMiner(distance=distance, **params["kwargs"])
        if params["class"] == "MultiSimilarityMiner":
            return MultiSimilarityMiner(distance=distance, **params["kwargs"])
        if params["class"] == "DistanceWeightedMiner":
            return DistanceWeightedMiner(distance=distance, **params["kwargs"])

        raise ValueError(f"Unknown miner: {params['class']}")

    def _setup_sampler(self):
        if "sampler" not in self.config["train_params"]:
            return None

        params = self.config["train_params"]["sampler"]
        if params["class"] == "MPerClassSampler":
            return MPerClassSampler(self.train_data.labels, **params["kwargs"])

        raise ValueError(f"Unknown sampler: {params['class']}")
