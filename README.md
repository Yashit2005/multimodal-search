# Multi-Modal Image Search — CLIP + FAISS

[![Live Demo](https://img.shields.io/badge/🔍%20Live%20Demo-GitHub%20Pages-6378ff?style=for-the-badge)](https://yashit2005.github.io/multimodal-search)
[![GitHub](https://img.shields.io/badge/GitHub-Yashit2005-181717?style=for-the-badge&logo=github)](https://github.com/Yashit2005/multimodal-search)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **[🚀 Interactive Demo →](https://yashit2005.github.io/multimodal-search)** — Run CLIP directly in your browser. Text search, image search, hybrid search, and zero-shot classification — no API key, no backend.

Search a corpus of images using **natural language**, an **example image**, or both simultaneously.
Built on OpenCLIP (ViT-B/32) and FAISS, served via FastAPI.

---

## What it does

| Query type | Example | How it works |
|---|---|---|
| Text → Image | *"a dog playing on the beach"* | Encode text with CLIP, find nearest image embeddings in FAISS |
| Image → Image | Upload a photo | Encode image with CLIP, find visually similar images |
| Hybrid | *"like this photo, but at sunset"* | Linearly blend text + image embeddings, search |
| Zero-shot classify | Upload image + label list | Softmax over image-text dot products |

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │  CLIP (ViT-B/32) encoder         │
                    │  openai trained on 400M pairs     │
                    └──────────┬──────────┬────────────┘
                               │          │
                    ┌──────────▼──┐  ┌───▼──────────┐
                    │ Text encoder│  │ Image encoder │
                    │  512-d emb  │  │  512-d emb    │
                    └──────────┬──┘  └───┬───────────┘
                               │         │
                               └────┬────┘
                                    │  same 512-d space
                                    ▼
                         ┌──────────────────┐
                         │  FAISS IndexFlat │  exact cosine similarity
                         │  N image vectors │  search in <5ms
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │  FastAPI serving  │
                         │  /search/text     │
                         │  /search/image    │
                         │  /search/hybrid   │
                         │  /classify        │
                         └──────────────────┘
```

---

## CLIP — how contrastive training works

CLIP is trained on 400 million (image, caption) pairs scraped from the web.
The training objective is **contrastive loss**:

- For a batch of N pairs, the model must match each image to its correct caption (N matches out of N² possible pairs).
- Image encoder and text encoder are trained jointly so that matching pairs have high cosine similarity and mismatched pairs have low similarity.
- After training, the embedding space is **modality-agnostic**: a photo of a dog and the text "a dog" land near each other, enabling cross-modal search.

**Zero-shot classification** falls out for free: embed the image, embed each class label as `"a photo of a {label}"`, take argmax of similarities.

---

## Key design decisions (interview-ready answers)

**Why cosine similarity via inner product on normalised vectors?**
Cosine similarity = inner product when both vectors are L2-normalised. FAISS `IndexFlatIP` (inner product) then gives exact cosine search with no extra computation.

**Why FAISS over a simple numpy search?**
Both give exact results for flat indices, but FAISS is implemented in C++ with SIMD optimisation. For 100K+ vectors, FAISS is 10-100× faster than numpy matmul.

**When would you switch from `IndexFlatIP` to `IndexHNSWFlat`?**
At ~1M+ vectors, exact search becomes slow (~100ms). HNSW gives ~95% recall in <5ms by building a graph of approximate neighbours. Trade: ~2× memory, one-time build cost.

**Why `ViT-B/32` over `ViT-L/14`?**
B/32 gives 512-d embeddings and runs at ~200 images/sec on CPU. L/14 gives 768-d and ~20× better quality but 10× slower. For a portfolio project B/32 is the right default — swap by changing one line.

**What is the hybrid search alpha parameter?**
`combined = α × text_emb + (1-α) × image_emb`. At α=1 it's pure text search; at α=0 it's pure image search. At α=0.5 both modalities contribute equally. This lets users say "find images like this photo, but with sunset lighting."

---

## Results

On 5K Flickr30K images, text-to-image retrieval:

| Metric | Score |
|---|---|
| Recall@1 | ~0.31 |
| Recall@5 | ~0.58 |
| Recall@10 | ~0.68 |

vs BM25 on captions: Recall@10 ≈ 0.28 (BM25 fails on semantic queries with no exact word overlap)

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/<you>/multimodal-search.git
cd multimodal-search
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Quick demo (no dataset download)

```bash
python scripts/demo.py
# Shows text similarity, zero-shot classification, image-text alignment
```

### 3. Build the full index (downloads Flickr30K via HuggingFace)

```bash
python pipeline.py --stage build --n-images 5000
# First run downloads CLIP weights (~350MB) and Flickr30K images
# Takes ~10 min on CPU, ~2 min on GPU
```

### 4. Serve

```bash
python pipeline.py --stage serve
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### 5. Search

```bash
# Text search
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "a dog playing on the beach", "top_k": 5}'

# Zero-shot classification
curl -X POST http://localhost:8000/classify/zero-shot \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "<base64>", "labels": ["cat", "dog", "bird", "horse"]}'

# Image-to-image search
curl -X POST http://localhost:8000/search/image \
  -F "file=@my_photo.jpg" -F "top_k=10"
```

### 6. BM25 vs CLIP analysis

```bash
pip install rank-bm25
python pipeline.py --stage analyze
# Generates reports/analysis/bm25_clip_overlap.png + tsne_embeddings.png
```

### 7. Run tests

```bash
pytest tests/ -v
```

---

## Project structure

```
multimodal-search/
├── pipeline.py                    # Main orchestrator
├── src/
│   ├── embeddings/
│   │   ├── clip_encoder.py        # CLIPEncoder: encode_image, encode_text, zero_shot_classify
│   │   └── build_index.py         # Download dataset, build FAISS index, evaluate Recall@K
│   ├── search/
│   │   └── index.py               # ImageIndex: add, search, text_search, hybrid_search, save/load
│   ├── serving/
│   │   └── api.py                 # FastAPI: /search/text, /search/image, /search/hybrid, /classify
│   └── utils/
│       └── analysis.py            # BM25 vs CLIP comparison, t-SNE visualisation, similarity heatmap
├── scripts/
│   └── demo.py                    # Quick demo — no index required
├── tests/
│   ├── test_encoder.py            # Unit tests for CLIPEncoder and ImageIndex
│   └── test_api.py                # API endpoint tests
├── Dockerfile
└── requirements.txt
```

---

## Interview talking points

- **"CLIP embeds images and text into the same 512-d space via contrastive training on 400M pairs. Cross-modal search works because matching (image, text) pairs are pulled together during training."**
- **"I use L2-normalised embeddings and FAISS IndexFlatIP — inner product on unit vectors equals cosine similarity, so I get exact cosine search without a separate normalisation step."**
- **"Zero-shot classification requires no labelled training data. I embed each class label as 'a photo of a {label}' and take argmax of image-text similarities. It works because CLIP saw these patterns during pretraining."**
- **"BM25 on captions gives Recall@10 ≈ 0.28. CLIP gives 0.68 — a 2.4× improvement — because CLIP captures semantic similarity rather than requiring exact word matches."**
- **"For production at 1M+ images, I'd switch from IndexFlatIP (exact, O(N)) to IndexHNSWFlat (approximate, O(log N)) trading ~5% recall for 100× query speed."**

---

## License

MIT
