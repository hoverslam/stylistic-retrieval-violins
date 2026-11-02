from pathlib import Path
import argparse

from stylistic_retrieval.utils.config import Config
from stylistic_retrieval.utils.visuals import Image
from stylistic_retrieval.retriever import Retriever, save_retrieval_results
from stylistic_retrieval.dataset import Instrument, PaschRichtersDataset
from stylistic_retrieval.utils.io import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run instrument retrieval on a set of query images.")

    parser.add_argument("--config", required=True, type=str, help="Path to the config YAML file.")
    parser.add_argument("--model", required=True, type=str, help="Path to the trained model checkpoint.")
    parser.add_argument("--input_dir", required=True, type=Path, help="Directory containing query images.")
    parser.add_argument(
        "--output_dir", required=True, type=Path, help="Directory where retrieval results will be saved."
    )
    parser.add_argument("--top_k", default=10, type=int, help="Number of retrieval results per query.")
    parser.add_argument("--device", default="cuda:0", type=str, help="Device to run inference on.")

    return parser.parse_args()


def create_queries(image_dir: Path) -> list[Instrument]:
    """Create query samples and corresponding Instrument objects from images.

    Args:
        image_dir: Directory containing query images.

    Returns:
        List of tuples containing:
            - image array
            - corresponding Instrument object
    """
    results: list[Instrument] = []
    aoi_data = read_json(image_dir / "areas_of_interest.json")

    for img_path in image_dir.iterdir():
        if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            img = Image(
                img_path,
                aois=aoi_data.get(img_path.name) if aoi_data else [],
            )

            instrument = Instrument(
                id=str(img_path.stem),
                maker="",
                place_of_production="",
                date_of_production="",
                images=[img],
            )

            results.append(instrument)

    return results


def main() -> None:
    args = parse_args()

    config = Config(file_path=args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = config["dataset"]

    # Load gallery dataset
    full_dataset = PaschRichtersDataset(
        root=dataset_config["dir"],
        meta_file=dataset_config["meta_file"],
        aoi_file=dataset_config["aoi_file"],
    )

    retriever = Retriever(full_dataset, args.model, config, device=args.device)

    # Create queries
    query_instruments = create_queries(args.input_dir)

    # Run retrieval
    results = retriever.retrieve(query_instruments, top_k=args.top_k)

    # Save results
    save_retrieval_results(query_instruments, results, output_dir)


if __name__ == "__main__":
    main()
