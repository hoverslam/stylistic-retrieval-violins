# Stylistic Retrieval of Historical Violins

Given photos of a violin, the system finds the instruments in a collection that look most stylistically similar to it. We treat instruments by the same maker as stylistically related, so the models are trained to place violins by the same maker close together in an embedding space. You can then retrieve them by nearest-neighbour search.

The models don't have to work on whole photos. They can focus on individual parts of the violin, each stored as a bounding box (*area of interest*, AOI) in the dataset:
- the whole instrument from the front, back or side (`violin-front`, `violin-back`, `violin-profile`)
- the top and back plates (`top-plate`, `back-plate`)
- the f-holes (`f-hole`)
- the C-bouts (`c-bout`, `c-bout-plus`)
- the scroll from the front, back or side (`scroll-front`, `scroll-back`, `scroll-profile`)

This makes it possible to compare which parts of a violin carry the most information about its maker.

## Data availability

The main dataset (Pasch/Richters, about 1,600 violins by 629 makers) is **proprietary and cannot be published**, so the dataset itself is not part of this repository.

A smaller, publicly available dataset will be published on Kaggle. A link and results on that dataset will follow.

## Installation

Requirements: Linux or Windows with an NVIDIA GPU (CUDA 12.4 wheels), and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv (Linux / macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell): powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone the repository
git clone https://github.com/hoverslam/stylistic-retrieval
cd stylistic-retrieval

# 3. Create the environment and install the dependencies
uv sync
```

`uv sync` uses the Python version pinned in `.python-version` (3.12). Use Python 3.10–3.12: `xformers` publishes no prebuilt wheels for newer versions, so uv would try to build it from source.

To check that PyTorch can see the GPU:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Dataset format

A dataset is a directory containing the images and a `meta/` folder:

```
<dataset_root>/
├── images/...                    # any structure; paths are referenced in instruments.json
└── meta/
    ├── instruments.json          # one entry per instrument
    ├── areas_of_interest.json    # bounding boxes of violin parts per image
    └── splits.json               # instrument ID -> 0 (train), 1 (validation), 2 (test)
```

`instruments.json`:

```json
[
  {
    "id": "X001",
    "maker": "Michael Alban",
    "place_of_production": "Graz",
    "date_of_production": "1707",
    "images": ["images/0/X001/000_X001-01.jpg", "..."]
  }
]
```

`areas_of_interest.json` maps each image path to a list of boxes. The coordinates are `[x_min, y_min, x_max, y_max]`:

```json
{
  "images/0/X001/000_X001-01.jpg": [
    {"label": "f-hole", "confidence": 0.98, "coordinates": [1513, 1620, 1902, 2273]}
  ]
}
```

Point the `dataset.dir` entry of a config to `<dataset_root>`. The `dataset.filters.aois` entry selects which violin parts are used.

## Usage

Each experiment is defined by one YAML config in `configs/`. Config names follow the pattern `<dataset>_<part>_<model>_<head>`, e.g. `configs/pasch_richters/dino/pr_scroll_dinov2_vits14_cls.yaml`.

```bash
# Train (logs to Weights & Biases; use WANDB_MODE=offline to keep the logs local)
uv run python scripts/train.py --config <config.yaml> --device 0

# Evaluate a checkpoint on train/val/test and write the metrics to an Excel file
uv run python scripts/evaluate.py -c <config.yaml> -m <checkpoint.pt> -o results.xlsx

# Retrieve the most similar instruments for a folder of query images
uv run python scripts/retrieve.py --config <config.yaml> --model <checkpoint.pt> --input_dir <query_dir> --top_k 10
```

Evaluation first averages the image embeddings of each instrument into one vector. It then reports retrieval metrics for finding instruments by the same maker: mAP, Top-1 and Hard/Soft/Precision@K.

## Models

All models map an image (or an AOI crop) to an L2-normalized embedding. They are trained with metric-learning losses from [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning), such as Multi-Similarity, Triplet, Contrastive, SupCon, ProxyAnchor and ArcFace. Training batches are built with `MPerClassSampler`, which puts several images of each maker in a batch. The model is selected with `model.architecture` in the config.

| Architecture | Backbone | Description |
|---|---|---|
| `ResNet` | ResNet-18/34/50/101/152 (ImageNet) | CNN baseline with GAP, GeM or NetVLAD pooling and an optional projection layer. |
| `DinoV2` | DINOv2 ViT-S/B/L/G (self-supervised) | Vision transformer that uses the CLS token or pooled patch tokens (GAP, GeM, NetVLAD, SemanticSpatialFusion). The backbone can be frozen for linear probing (`_lp` configs) or fine-tuned with LoRA. |
| `CricaVPR` | DINOv2 ViT-S/B/L with adapters | Adapted from visual place recognition. Small convolutional adapters inside each frozen transformer block, plus a cross-image encoder that lets the images in a batch attend to each other. ViT-B is the original configuration. |

**Note on CricaVPR:** because of its cross-image encoder, an image's embedding depends on the other images in the same batch. With `test_settings.batch_by_instrument: true`, evaluation and retrieval process one instrument per batch, so no information leaks between instruments. This is enabled in all CricaVPR configs.

## Repository structure

```
configs/               experiment configs (Pasch/Richters, MIMO)
scripts/               train, evaluate, retrieve, scrape MIMO
stylistic_retrieval/
├── dataset.py         dataset classes, AOI cropping, instrument batching
├── trainer.py         training loop with evaluation, early stopping and W&B logging
├── evaluator.py       feature extraction, optional PCA, retrieval metrics
├── retriever.py       gallery building and retrieval
├── scraper.py         scraper for MIMO (Musical Instrument Museums Online)
└── models/            ResNet, DINOv2, CricaVPR, aggregation layers
notebooks/             data exploration, clustering, visualizations
```

## Acknowledgements

- [DINOv2](https://github.com/facebookresearch/dinov2) (Oquab et al., 2023)
- [CricaVPR](https://github.com/Lu-Feng/CricaVPR) (Lu et al., CVPR 2024): the files in `stylistic_retrieval/models/cricavpr/` are adapted from this repository.

## License

MIT. See [LICENSE.txt](LICENSE.txt). The adapted CricaVPR and DINOv2 code keeps its original licenses, included in `stylistic_retrieval/models/cricavpr/`.
