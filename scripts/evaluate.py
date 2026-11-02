from stylistic_retrieval.utils.config import Config
from stylistic_retrieval.utils.preprocessing import setup_transform
from stylistic_retrieval.dataset import PaschRichtersDataset, TorchDatasetWrapper
from stylistic_retrieval.models.retrieval import setup_retrieval_model
from stylistic_retrieval.evaluator import RetrievalEvaluator

import argparse
from pathlib import Path

import torch
import pandas as pd


def load_model(model_path: Path | str, config: Config, device: str) -> torch.nn.Module:
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at '{model_path.as_posix()}'")

    model = setup_retrieval_model(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation based on a configuration file.")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to the trained model weights file.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the CSV output file.")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Device to run training on.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = Config(args.config)
    model = load_model(model_path=args.model, config=config, device=args.device)

    test_transform = setup_transform(config, mode="test")

    dataset_config = config["dataset"]
    full_dataset = PaschRichtersDataset(
        root=dataset_config["dir"],
        meta_file=dataset_config["meta_file"],
        aoi_file=dataset_config["aoi_file"],
    )
    train_split, val_split, test_split = full_dataset.split(dataset_config["split_file"])

    evaluator = RetrievalEvaluator(model, config, args.device, num_workers=1)

    splits = {
        "train": train_split,
        "val": val_split,
        "test": test_split,
    }

    raw_sheets = {}
    summary_rows = []
    for split_name, split_data in splits.items():
        data = TorchDatasetWrapper(
            split_data,
            dataset_config.get("filters", {}),
            transform=test_transform,
        )

        split_result = evaluator.evaluate(data, prefix=f"{split_name}_")

        # raw
        raw_key = f"{split_name}_raw"
        raw_sheets[raw_key] = pd.DataFrame(split_result.get(raw_key, {}))

        # metrics
        for key, value in split_result.items():
            if key.startswith(f"{split_name}_") and key != f"{split_name}_raw":
                metric = key.replace(f"{split_name}_", "")
                summary_rows.append({"metric": metric, "split": split_name, "value": value})

    summary_df = pd.DataFrame(summary_rows).pivot(index="metric", columns="split", values="value")

    # ---- write Excel ----
    output_path = Path(args.output)
    if output_path.suffix.lower() != ".xlsx":
        output_path = output_path.with_suffix(".xlsx")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary")

        for sheet_name, df in raw_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    print(f"\nSaved evaluation to: {output_path}")


if __name__ == "__main__":
    main()
