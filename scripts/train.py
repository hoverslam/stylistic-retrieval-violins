from stylistic_retrieval.dataset import MIMODataset, TorchDatasetWrapper
from stylistic_retrieval.models.retrieval import setup_retrieval_model
from stylistic_retrieval.utils.config import Config
from stylistic_retrieval.utils.preprocessing import setup_transform
from stylistic_retrieval.trainer import RetrievalTrainer

import os
import logging
import argparse
from argparse import Namespace
from pathlib import Path

import torch

# This fixes 90% of OpenBLAS errors
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Logger settings
log_dir = Path(".temp")
log_dir.mkdir(exist_ok=True)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "training.log"),
        logging.StreamHandler(),
    ],
)


def get_args() -> Namespace:
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(description="Run training based on a configuration file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers. (Default: 8)")
    parser.add_argument(
        "--device",
        type=int,
        nargs="+",
        default=[0],
        help="List of GPU IDs for training (e.g., --device 0 1 2). (Default: [0])",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution script for training the retrieval model."""
    # --- 1. Configuration and Arguments ---
    args = get_args()
    config = Config(args.config)

    # --- 2. Data and Transformations ---
    print("Setting up datasets and transformations...")
    train_transform = setup_transform(config, mode="train")
    test_transform = setup_transform(config, mode="test")

    dataset_config = config["dataset"]
    full_dataset = MIMODataset(
        root=dataset_config["dir"],
        meta_file=dataset_config["meta_file"],
        aoi_file=dataset_config["aoi_file"],
    )
    train_split, val_split, _ = full_dataset.split(dataset_config["split_file"])

    train_data = TorchDatasetWrapper(train_split, dataset_config.get("filters", {}), transform=train_transform)
    eval_data = {
        "train": TorchDatasetWrapper(train_split, dataset_config.get("filters", {}), transform=test_transform),
        "val": TorchDatasetWrapper(val_split, dataset_config.get("filters", {}), transform=test_transform),
    }

    # --- 3. Model Initialization ---
    print("Initializing model...")
    model = setup_retrieval_model(config)
    primary_device = f"cuda:{args.device[0]}"
    if len(args.device) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device)

    model.to(primary_device)

    # --- 4. Trainer Initialization and Execution ---
    print("Setting up trainer...")
    trainer = RetrievalTrainer(
        model=model,
        train_data=train_data,
        eval_data=eval_data,
        config=config,
        device=primary_device,
        num_workers=args.num_workers,
    )
    trainer.run()


if __name__ == "__main__":
    main()
