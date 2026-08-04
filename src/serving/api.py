"""
FastAPI serving layer for multi-modal search.

Endpoints:
  GET  /health                 — liveness check
  GET  /index/stats            — index size, model info
  POST /search/text            — text-to-image search
  POST /search/image           — image-to-image search
  POST /search/hybrid          — combined text + image query
  POST /classify/zero-shot     — zero-shot image classification
  POST /embed/text             — raw text embedding (for debugging / analysis)
  POST /embed/image            — raw image embedding
"""

import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
state: dict = {}

INDEX_DIR  = os.getenv("INDEX_DIR",  "data/index")
MODEL_NAME = os.getenv("MODEL_NAME", "ViT-B-32")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.embeddings.clip_encoder import CLIPEncoder
    from src.search.index import ImageIndex

    logger.info(f"Loading CLIP model: {MODEL_NAME}")
    state["encoder"] = CLIPEncoder(model_name=MODEL_NAME)

    logger.info(f"Loading FAISS index from: {INDEX_DIR}")
    state["index"] = ImageIndex.load(INDEX_DIR)
    state["loaded_at"] = time.time()
    logger.info("Ready.")
    yield
    state.clear()


app = FastAPI(
    title="Multi-Modal Search API",
    description="Search images by text, by example image, or both (CLIP + FAISS)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pil_from_upload(upload: UploadFile) -> Image.Image:
    return Image.open(io.BytesIO(upload.file.read())).convert("RGB")


def pil_from_base64(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def format_results(results: list[dict]) -> list[dict]:
    """Clean up search results for API response."""
    out = []
    for r in results:
        out.append({
            "rank":       r["rank"],
            "score":      round(r["score"], 4),
            "image_id":   r.get("image_id", ""),
            "image_path": r.get("image_path", ""),
            "caption":    r.get("caption", ""),
        })
    return out


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TextSearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(10, ge=1, le=100)

    class Config:
        json_schema_extra = {"example": {"query": "a dog playing on the beach", "top_k": 5}}


class HybridSearchRequest(BaseModel):
    query: str = Field(..., description="Text query")
    image_b64: str = Field(..., description="Base64-encoded image")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="1.0=text only, 0.0=image only")
    top_k: int = Field(10, ge=1, le=100)


class ZeroShotRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded image")
    labels: list[str] = Field(..., description="Candidate class labels")
    prompt_template: str = Field("a photo of a {}", description="Prompt wrapping each label")

    class Config:
        json_schema_extra = {
            "example": {
                "image_b64": "<base64>",
                "labels": ["cat", "dog", "bird", "horse"],
                "prompt_template": "a photo of a {}",
            }
        }


class EmbedTextRequest(BaseModel):
    texts: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "encoder" in state,
        "index_loaded": "index" in state,
        "index_size": len(state["index"]) if "index" in state else 0,
    }


@app.get("/index/stats")
def index_stats():
    if "index" not in state:
        raise HTTPException(503, "Index not loaded")
    idx = state["index"]
    return {
        "n_images": len(idx),
        "embed_dim": idx.embed_dim,
        "index_type": idx.index_type,
        "model": MODEL_NAME,
    }


@app.post("/search/text")
def search_text(req: TextSearchRequest):
    """Search images by natural language query."""
    if "encoder" not in state:
        raise HTTPException(503, "Model not loaded")
    t0 = time.time()
    emb = state["encoder"].encode_text(req.query)
    results = state["index"].text_search(emb, top_k=req.top_k)
    return {
        "query": req.query,
        "results": format_results(results),
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }


@app.post("/search/image")
async def search_image(file: UploadFile = File(...), top_k: int = Form(10)):
    """Search for visually similar images by uploading an example image."""
    if "encoder" not in state:
        raise HTTPException(503, "Model not loaded")
    t0 = time.time()
    img = pil_from_upload(file)
    emb = state["encoder"].encode_image(img)
    results = state["index"].image_search(emb, top_k=top_k)
    return {
        "results": format_results(results),
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }


@app.post("/search/hybrid")
def search_hybrid(req: HybridSearchRequest):
    """
    Combine a text query and a reference image.
    alpha controls the blend: 1.0 = pure text, 0.0 = pure image.
    Useful for 'find images like this, but with sunset lighting'.
    """
    if "encoder" not in state:
        raise HTTPException(503, "Model not loaded")
    t0 = time.time()
    text_emb = state["encoder"].encode_text(req.query)
    img = pil_from_base64(req.image_b64)
    img_emb = state["encoder"].encode_image(img)
    results = state["index"].hybrid_search(text_emb, img_emb, alpha=req.alpha, top_k=req.top_k)
    return {
        "query": req.query,
        "alpha": req.alpha,
        "results": format_results(results),
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }


@app.post("/classify/zero-shot")
def zero_shot_classify(req: ZeroShotRequest):
    """
    Classify an image into provided categories — no training required.

    How: embed each label as text, embed the image, softmax over dot products.
    Works for any set of categories you define at query time.
    """
    if "encoder" not in state:
        raise HTTPException(503, "Model not loaded")
    if len(req.labels) > 100:
        raise HTTPException(400, "Max 100 labels")
    t0 = time.time()
    img = pil_from_base64(req.image_b64)
    probs = state["encoder"].zero_shot_classify(img, req.labels, req.prompt_template)[0]
    sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
    return {
        "predictions": [{"label": k, "probability": round(v, 4)} for k, v in sorted_probs],
        "top_label": sorted_probs[0][0],
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }


@app.post("/embed/text")
def embed_text(req: EmbedTextRequest):
    """Return raw text embeddings (useful for analysis / debugging)."""
    if "encoder" not in state:
        raise HTTPException(503, "Model not loaded")
    if len(req.texts) > 100:
        raise HTTPException(400, "Max 100 texts per call")
    embs = state["encoder"].encode_text(req.texts)
    return {
        "embeddings": embs.tolist(),
        "shape": list(embs.shape),
        "model": MODEL_NAME,
    }


@app.post("/embed/image")
async def embed_image(file: UploadFile = File(...)):
    """Return raw image embedding."""
    if "encoder" not in state:
        raise HTTPException(503, "Model not loaded")
    img = pil_from_upload(file)
    emb = state["encoder"].encode_image(img)
    return {
        "embedding": emb[0].tolist(),
        "shape": list(emb.shape),
        "model": MODEL_NAME,
    }
