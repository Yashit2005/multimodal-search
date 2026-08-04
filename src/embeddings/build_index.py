"""
Index builder — downloads a dataset and builds the FAISS index.

Supported datasets:
  - flickr30k:  ~31K images with 5 captions each (recommended for portfolio)
  - coco-val:   ~5K COCO validation images with captions
  - custom:     point at a folder of images

Usage:
    python -m src.embeddings.build_index --dataset flickr30k --output data/index/
    python -m src.embeddings.build_index --dataset custom --images-dir data/my_photos/
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.embeddings.clip_encoder import CLIPEncoder
from src.search.index import ImageIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_flickr30k_sample(data_dir: str, n_images: int = 5000) -> list[dict]:
    """
    Load a sample of Flickr30K from Hugging Face datasets.

    Each record: {image_id, image_path (saved locally), caption, split}
    """
    from datasets import load_dataset

    logger.info(f"Downloading Flickr30K (first {n_images} images)...")
    ds = load_dataset("nlphuji/flickr30k", split="test", streaming=True)

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    records = []

    for i, item in enumerate(tqdm(ds, total=n_images, desc="Saving images")):
        if i >= n_images:
            break
        img_path = f"{data_dir}/{i:05d}.jpg"
        item["image"].save(img_path, format="JPEG")
        records.append({
            "image_id": str(i),
            "image_path": img_path,
            "caption": item["caption"][0] if isinstance(item["caption"], list) else item["caption"],
            "all_captions": item["caption"] if isinstance(item["caption"], list) else [item["caption"]],
            "split": "test",
        })

    logger.info(f"Saved {len(records)} Flickr30K images to {data_dir}/")
    return records


def load_coco_sample(data_dir: str, n_images: int = 2000) -> list[dict]:
    """Load COCO 2017 validation subset from Hugging Face."""
    from datasets import load_dataset
    logger.info(f"Downloading COCO val (first {n_images} images)...")
    ds = load_dataset("detection-datasets/coco", split="val", streaming=True)

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    records = []
    seen = set()

    for item in tqdm(ds, total=n_images, desc="Saving COCO images"):
        if len(records) >= n_images:
            break
        img_id = str(item["image_id"])
        if img_id in seen:
            continue
        seen.add(img_id)
        img_path = f"{data_dir}/{img_id}.jpg"
        item["image"].save(img_path, format="JPEG")
        records.append({
            "image_id": img_id,
            "image_path": img_path,
            "caption": f"An image from COCO dataset (id={img_id})",
            "categories": [a["category_id"] for a in item.get("annotations", [])],
        })

    logger.info(f"Saved {len(records)} COCO images.")
    return records


def load_custom_images(images_dir: str) -> list[dict]:
    """Load all images from a local directory."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = [p for p in Path(images_dir).rglob("*") if p.suffix.lower() in exts]
    logger.info(f"Found {len(paths)} images in {images_dir}")
    return [
        {
            "image_id": p.stem,
            "image_path": str(p),
            "caption": p.stem.replace("_", " ").replace("-", " "),
        }
        for p in paths
    ]


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(
    records: list[dict],
    encoder: CLIPEncoder,
    output_dir: str,
    batch_size: int = 64,
    also_encode_captions: bool = True,
) -> ImageIndex:
    """
    Embed all images (and optionally captions) and add to FAISS index.

    Caption embeddings are stored separately — used for hybrid search
    and for evaluating retrieval metrics.
    """
    index = ImageIndex(embed_dim=encoder.embed_dim)
    caption_embeddings = []
    image_embeddings = []

    logger.info(f"Encoding {len(records)} images...")
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        paths = [r["image_path"] for r in batch]

        try:
            embs = encoder.encode_image(paths)
        except Exception as e:
            logger.warning(f"Batch {i//batch_size} failed: {e} — skipping")
            continue

        index.add(embs, batch)
        image_embeddings.append(embs)

        if also_encode_captions:
            captions = [r.get("caption", "") for r in batch]
            cap_embs = encoder.encode_text(captions)
            caption_embeddings.append(cap_embs)

    # Save index
    index.save(output_dir)

    # Save embeddings as numpy arrays for evaluation / analysis
    np_dir = f"{output_dir}/embeddings"
    Path(np_dir).mkdir(exist_ok=True)
    np.save(f"{np_dir}/image_embeddings.npy", np.concatenate(image_embeddings))
    if caption_embeddings:
        np.save(f"{np_dir}/caption_embeddings.npy", np.concatenate(caption_embeddings))

    # Save metadata as parquet for easy inspection
    pd.DataFrame(records[:index.index.ntotal]).to_parquet(f"{output_dir}/metadata.parquet", index=False)
    logger.info(f"Index built: {len(index)} images in {output_dir}/")
    return index


# ---------------------------------------------------------------------------
# Evaluation: recall@K
# ---------------------------------------------------------------------------

def evaluate_retrieval(index: ImageIndex, encoder: CLIPEncoder, records: list[dict]) -> dict:
    """
    Evaluate text-to-image retrieval using caption as query.

    For each image, query with its own caption — the ground-truth match
    should appear in the top-K results.

    Metric: Recall@K (what fraction of queries retrieve the correct image in top K)
    """
    logger.info("Evaluating retrieval Recall@K...")
    hits = {1: 0, 5: 0, 10: 0}
    n = 0

    for record in tqdm(records[:500], desc="Evaluating"):  # 500-image sample
        caption = record.get("caption", "")
        if not caption:
            continue

        text_emb = encoder.encode_text(caption)
        results = index.text_search(text_emb, top_k=10)
        retrieved_ids = [r["image_id"] for r in results]

        for k in [1, 5, 10]:
            if record["image_id"] in retrieved_ids[:k]:
                hits[k] += 1
        n += 1

    metrics = {f"recall@{k}": round(hits[k] / n, 4) for k in [1, 5, 10]}
    logger.info(f"Retrieval metrics (n={n}): {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["flickr30k", "coco", "custom"], default="flickr30k")
    parser.add_argument("--images-dir", default="data/raw/images")
    parser.add_argument("--n-images", type=int, default=5000)
    parser.add_argument("--output", default="data/index")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    encoder = CLIPEncoder(model_name=args.model)

    if args.dataset == "flickr30k":
        records = load_flickr30k_sample(args.images_dir, args.n_images)
    elif args.dataset == "coco":
        records = load_coco_sample(args.images_dir, args.n_images)
    else:
        records = load_custom_images(args.images_dir)

    index = build_index(records, encoder, args.output)
    print(f"\nIndex: {index}")

    if args.evaluate:
        metrics = evaluate_retrieval(index, encoder, records)
        print(f"\nRetrieval metrics: {metrics}")
        Path(f"{args.output}/eval_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
