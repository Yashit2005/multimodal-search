"""
FAISS index — stores image embeddings and enables fast nearest-neighbour search.

Index types (tradeoffs matter for interviews):
  - IndexFlatIP:   exact brute-force inner product (small corpora, <100K images)
  - IndexIVFFlat:  inverted file index — clusters vectors, only searches nearby clusters
                   Faster but approximate. Needs training. Good for 100K–10M images.
  - IndexHNSWFlat: graph-based approximate search. No training, very fast, high recall.
                   Best for production without a GPU.

We use IndexFlatIP (exact) by default since correctness > speed for a portfolio project,
but the code makes switching trivial.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageIndex:
    """
    FAISS-backed image search index.

    Stores embeddings + metadata. Supports:
      - Building from scratch
      - Incremental adds
      - Save / load from disk
      - Text-to-image, image-to-image, and hybrid search
    """

    def __init__(self, embed_dim: int = 512, index_type: str = "flat"):
        self.embed_dim = embed_dim
        self.index_type = index_type
        self.index = self._build_index(embed_dim, index_type)
        self.metadata: list[dict] = []   # parallel list to FAISS ids

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self, dim: int, index_type: str) -> faiss.Index:
        if index_type == "flat":
            # Exact inner product (cosine sim on normalised vectors)
            return faiss.IndexFlatIP(dim)
        elif index_type == "ivf":
            # Approximate — needs training on sample vectors
            quantiser = faiss.IndexFlatIP(dim)
            return faiss.IndexIVFFlat(quantiser, dim, 100, faiss.METRIC_INNER_PRODUCT)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 128
            return index
        else:
            raise ValueError(f"Unknown index type: {index_type}. Choose: flat, ivf, hnsw")

    def add(
        self,
        embeddings: np.ndarray,
        metadata: list[dict],
    ) -> None:
        """
        Add embeddings + metadata to the index.

        Args:
            embeddings: float32 array of shape (N, embed_dim), L2-normalised
            metadata:   list of N dicts with image info (path, caption, id, etc.)
        """
        assert embeddings.shape[1] == self.embed_dim, (
            f"Embedding dim mismatch: got {embeddings.shape[1]}, expected {self.embed_dim}"
        )
        assert len(embeddings) == len(metadata)

        embeddings = embeddings.astype(np.float32)

        # IVF needs training before first add
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)

        self.index.add(embeddings)
        self.metadata.extend(metadata)
        logger.info(f"Index now contains {self.index.ntotal} vectors")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_fn=None,
    ) -> list[dict]:
        """
        Find top-k nearest neighbours.

        Args:
            query_embedding: shape (1, D) or (D,) — text or image embedding
            top_k:           number of results
            filter_fn:       optional callable(metadata_dict) -> bool for post-filtering

        Returns:
            list of dicts: {rank, score, **metadata_fields}
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[np.newaxis, :]
        query_embedding = query_embedding.astype(np.float32)

        # Fetch more than top_k if we need to filter
        fetch_k = min(top_k * 5 if filter_fn else top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            if filter_fn and not filter_fn(meta):
                continue
            results.append({"rank": len(results) + 1, "score": float(score), **meta})
            if len(results) == top_k:
                break

        return results

    def text_search(self, text_embedding: np.ndarray, top_k: int = 10) -> list[dict]:
        """Text-to-image search."""
        return self.search(text_embedding, top_k)

    def image_search(self, image_embedding: np.ndarray, top_k: int = 10) -> list[dict]:
        """Image-to-image search (find visually similar images)."""
        return self.search(image_embedding, top_k)

    def hybrid_search(
        self,
        text_embedding: np.ndarray,
        image_embedding: np.ndarray,
        alpha: float = 0.5,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Combine text and image queries linearly.

        alpha=1.0 → pure text query
        alpha=0.0 → pure image query
        alpha=0.5 → equal weight (default)

        This is useful when user provides both a reference image and a text
        description: e.g. "find images like this photo, but with sunset lighting"
        """
        combined = alpha * text_embedding + (1 - alpha) * image_embedding
        norm = np.linalg.norm(combined, axis=-1, keepdims=True)
        combined = combined / np.maximum(norm, 1e-8)
        return self.search(combined, top_k)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{directory}/index.faiss")
        with open(f"{directory}/metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        with open(f"{directory}/config.json", "w") as f:
            json.dump({"embed_dim": self.embed_dim, "index_type": self.index_type,
                        "n_vectors": self.index.ntotal}, f)
        logger.info(f"Index saved to {directory}/ ({self.index.ntotal} vectors)")

    @classmethod
    def load(cls, directory: str) -> "ImageIndex":
        with open(f"{directory}/config.json") as f:
            cfg = json.load(f)
        obj = cls.__new__(cls)
        obj.embed_dim = cfg["embed_dim"]
        obj.index_type = cfg["index_type"]
        obj.index = faiss.read_index(f"{directory}/index.faiss")
        with open(f"{directory}/metadata.pkl", "rb") as f:
            obj.metadata = pickle.load(f)
        logger.info(f"Loaded index: {obj.index.ntotal} vectors from {directory}/")
        return obj

    def __len__(self):
        return self.index.ntotal

    def __repr__(self):
        return (f"ImageIndex(type={self.index_type}, dim={self.embed_dim}, "
                f"n_vectors={self.index.ntotal})")
