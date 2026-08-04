"""
Analysis utilities:
  1. BM25 vs CLIP retrieval comparison (the key interview story)
  2. t-SNE / UMAP embedding space visualisation
  3. Similarity heatmap across query set

Run standalone:
    python -m src.utils.analysis --index data/index/ --mode compare
    python -m src.utils.analysis --index data/index/ --mode visualize
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.manifold import TSNE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BM25 retrieval (keyword baseline)
# ---------------------------------------------------------------------------

def build_bm25_index(captions: list[str]):
    """Build a BM25 index over image captions."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError("pip install rank-bm25")

    tokenised = [cap.lower().split() for cap in captions]
    return BM25Okapi(tokenised)


def bm25_search(bm25, captions: list[str], query: str, top_k: int = 10) -> list[dict]:
    scores = bm25.get_scores(query.lower().split())
    ranked = np.argsort(scores)[::-1][:top_k]
    return [
        {"rank": i + 1, "score": float(scores[idx]), "caption": captions[idx], "index": int(idx)}
        for i, idx in enumerate(ranked)
    ]


# ---------------------------------------------------------------------------
# Head-to-head: BM25 vs CLIP
# ---------------------------------------------------------------------------

def compare_bm25_vs_clip(
    index_dir: str,
    queries: list[str],
    top_k: int = 10,
    output_dir: str = "reports/analysis",
) -> pd.DataFrame:
    """
    For each query, retrieve top-K with BM25 and CLIP.
    Compute:
      - Score overlap (how many results appear in both top-K)
      - CLIP-BM25 score correlation

    Key finding to highlight in interviews:
      BM25 fails on semantic queries ("a joyful moment outdoors")
      because it requires exact word overlap with captions.
      CLIP retrieves semantically similar images even when no words match.
    """
    from src.embeddings.clip_encoder import CLIPEncoder
    from src.search.index import ImageIndex

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    encoder = CLIPEncoder()
    faiss_index = ImageIndex.load(index_dir)

    meta_df = pd.read_parquet(f"{index_dir}/metadata.parquet")
    captions = meta_df["caption"].fillna("").tolist()
    bm25 = build_bm25_index(captions)

    rows = []
    for query in queries:
        # CLIP search
        text_emb = encoder.encode_text(query)
        clip_results = faiss_index.text_search(text_emb, top_k=top_k)
        clip_indices = {r["image_id"] for r in clip_results}

        # BM25 search
        bm25_results = bm25_search(bm25, captions, query, top_k)
        bm25_indices = {str(r["index"]) for r in bm25_results}

        overlap = len(clip_indices & bm25_indices)
        rows.append({
            "query": query,
            "clip_top1_caption": clip_results[0]["caption"] if clip_results else "",
            "bm25_top1_caption": bm25_results[0]["caption"] if bm25_results else "",
            "top_k_overlap": overlap,
            "clip_top1_score": clip_results[0]["score"] if clip_results else 0,
            "bm25_top1_score": bm25_results[0]["score"] if bm25_results else 0,
        })
        logger.info(
            f"Query: '{query}' | Overlap: {overlap}/{top_k} | "
            f"CLIP: {clip_results[0]['caption'][:60] if clip_results else 'N/A'}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(f"{output_dir}/bm25_vs_clip.csv", index=False)

    # Plot overlap
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(queries)), df["top_k_overlap"], color="#185FA5", alpha=0.8)
    ax.set_xticks(range(len(queries)))
    ax.set_xticklabels([q[:30] for q in queries], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(f"Results in both top-{top_k}")
    ax.set_title(f"BM25 vs CLIP: top-{top_k} result overlap by query")
    ax.axhline(top_k * 0.5, color="gray", linestyle="--", alpha=0.5, label="50% overlap")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bm25_clip_overlap.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparison saved to {output_dir}/")
    return df


# ---------------------------------------------------------------------------
# t-SNE embedding space visualisation
# ---------------------------------------------------------------------------

def visualise_embedding_space(
    index_dir: str,
    n_samples: int = 1000,
    output_dir: str = "reports/analysis",
    color_by: str = "split",  # or any metadata field
):
    """
    Project image embeddings to 2D with t-SNE and plot.

    This is a strong portfolio demo: it shows that CLIP's embedding space
    clusters semantically related images together even with no labels.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    emb_path = f"{index_dir}/embeddings/image_embeddings.npy"
    meta_path = f"{index_dir}/metadata.parquet"

    embeddings = np.load(emb_path)
    meta_df = pd.read_parquet(meta_path)

    n = min(n_samples, len(embeddings))
    idx = np.random.choice(len(embeddings), n, replace=False)
    embs = embeddings[idx]
    meta = meta_df.iloc[idx].reset_index(drop=True)

    logger.info(f"Running t-SNE on {n} embeddings (this takes ~1 min)...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    coords = tsne.fit_transform(embs)

    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        s=8, alpha=0.6, c=range(n), cmap="tab20"
    )
    ax.set_title(f"CLIP embedding space — t-SNE ({n} images)")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.axis("off")
    plt.tight_layout()
    path = f"{output_dir}/tsne_embeddings.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"t-SNE plot saved to {path}")
    return coords


# ---------------------------------------------------------------------------
# Similarity heatmap
# ---------------------------------------------------------------------------

def similarity_heatmap(
    texts: list[str],
    encoder,
    output_dir: str = "reports/analysis",
):
    """
    Plot cosine similarity between a set of text queries.
    Reveals semantic structure of the embedding space.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    embs = encoder.encode_text(texts)
    sim_matrix = embs @ embs.T

    fig, ax = plt.subplots(figsize=(9, 8))
    import seaborn as sns
    sns.heatmap(
        sim_matrix,
        xticklabels=texts, yticklabels=texts,
        annot=True, fmt=".2f", cmap="Blues",
        vmin=0, vmax=1, ax=ax, annot_kws={"size": 8}
    )
    ax.set_title("CLIP text-text cosine similarity")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    path = f"{output_dir}/similarity_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap saved to {path}")
    return sim_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/index")
    parser.add_argument("--mode", choices=["compare", "visualize", "heatmap"], default="compare")
    parser.add_argument("--output", default="reports/analysis")
    args = parser.parse_args()

    DEMO_QUERIES = [
        "a dog playing in the park",
        "sunset over the ocean",
        "a group of people eating together",
        "a child riding a bicycle",
        "a joyful moment outdoors",      # semantic — BM25 struggles here
        "happiness and celebration",      # purely abstract — BM25 fails completely
        "urban street at night",
        "a red vehicle on a road",
    ]

    if args.mode == "compare":
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            print("Install rank-bm25: pip install rank-bm25")
            exit(1)
        compare_bm25_vs_clip(args.index, DEMO_QUERIES, output_dir=args.output)

    elif args.mode == "visualize":
        visualise_embedding_space(args.index, output_dir=args.output)

    elif args.mode == "heatmap":
        from src.embeddings.clip_encoder import CLIPEncoder
        encoder = CLIPEncoder()
        similarity_heatmap(DEMO_QUERIES[:6], encoder, output_dir=args.output)
