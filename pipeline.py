"""
Multi-modal search pipeline — main entry point.

Usage:
    python pipeline.py --stage build    # download Flickr30K + build FAISS index
    python pipeline.py --stage serve    # start FastAPI server
    python pipeline.py --stage analyze  # BM25 vs CLIP comparison + t-SNE
    python pipeline.py --stage demo     # quick demo without full index
    python pipeline.py                  # build → analyze

Download Flickr30K (auto via Hugging Face datasets, no Kaggle account needed):
    python pipeline.py --stage build --n-images 5000
"""

import argparse
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def stage_build(args):
    from src.embeddings.build_index import main as build_main
    sys.argv = [
        "build",
        "--dataset", args.dataset,
        "--images-dir", "data/raw/images",
        "--n-images", str(args.n_images),
        "--output", "data/index",
        "--model", args.model,
    ]
    if args.evaluate:
        sys.argv.append("--evaluate")
    build_main()


def stage_serve(args):
    cmd = [
        "uvicorn", "src.serving.api:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
    ]
    env = os.environ.copy()
    env["INDEX_DIR"] = args.index_dir
    env["MODEL_NAME"] = args.model
    logger.info(f"Starting server: http://localhost:8000")
    logger.info(f"API docs: http://localhost:8000/docs")
    subprocess.run(cmd, env=env)


def stage_analyze(args):
    from src.utils.analysis import compare_bm25_vs_clip, visualise_embedding_space

    queries = [
        "a dog playing in the park",
        "sunset over the ocean",
        "children playing outside",
        "a joyful moment outdoors",
        "happiness and celebration",
        "urban street at night",
    ]
    logger.info("Running BM25 vs CLIP comparison...")
    compare_bm25_vs_clip(args.index_dir, queries, output_dir="reports/analysis")

    logger.info("Running t-SNE visualisation...")
    visualise_embedding_space(args.index_dir, n_samples=500, output_dir="reports/analysis")
    logger.info("Reports saved to reports/analysis/")


def stage_demo(args):
    subprocess.run([sys.executable, "scripts/demo.py"])


def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Image Search Pipeline")
    parser.add_argument("--stage", choices=["build", "serve", "analyze", "demo", "all"],
                        default="all")
    parser.add_argument("--dataset", choices=["flickr30k", "coco", "custom"], default="flickr30k")
    parser.add_argument("--n-images", type=int, default=5000)
    parser.add_argument("--model", default="ViT-B-32",
                        help="CLIP model: ViT-B-32 (fast) | ViT-B-16 | ViT-L-14 (best)")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--evaluate", action="store_true",
                        help="Compute Recall@K after building")
    args = parser.parse_args()

    if args.stage in ("all", "build"):
        stage_build(args)
    if args.stage in ("all", "analyze"):
        stage_analyze(args)
    if args.stage == "serve":
        stage_serve(args)
    if args.stage == "demo":
        stage_demo(args)


if __name__ == "__main__":
    main()
