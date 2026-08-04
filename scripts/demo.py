"""
Quick demo — runs without the full indexed dataset.

Shows:
  1. Text embedding similarity between related/unrelated sentences
  2. Zero-shot image classification on a downloaded sample image
  3. Image-text similarity scoring

Run:
    python scripts/demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import requests
from io import BytesIO
from PIL import Image


def download_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=10)
    return Image.open(BytesIO(resp.content)).convert("RGB")


def main():
    print("Loading CLIP (ViT-B/32) — first run downloads ~350MB weights...")
    from src.embeddings.clip_encoder import CLIPEncoder
    encoder = CLIPEncoder(model_name="ViT-B-32")

    # ------------------------------------------------------------------
    # 1. Text similarity — the embedding space makes semantic sense
    # ------------------------------------------------------------------
    print("\n=== Text similarity ===")
    texts = [
        "a dog playing fetch",
        "a puppy running in the grass",   # semantically close to above
        "a cat sleeping on a sofa",       # different animal
        "a rocket launching into space",  # totally unrelated
    ]
    embs = encoder.encode_text(texts)
    sim = embs @ embs.T
    print(f"{'':35s}", end="")
    for t in texts:
        print(f"{t[:18]:20s}", end="")
    print()
    for i, t in enumerate(texts):
        print(f"{t[:35]:35s}", end="")
        for j in range(len(texts)):
            print(f"  {sim[i,j]:.3f}          ", end="")
        print()

    # ------------------------------------------------------------------
    # 2. Zero-shot classification on a real image
    # ------------------------------------------------------------------
    print("\n=== Zero-shot classification ===")
    # Wikimedia Commons — public domain dog image
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg"
    try:
        img = download_image(url)
        print(f"Downloaded image: {img.size}")
        labels = ["dog", "cat", "bird", "horse", "fish", "rabbit"]
        probs = encoder.zero_shot_classify(img, labels)[0]
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        print("Predictions:")
        for label, prob in sorted_probs:
            bar = "█" * int(prob * 40)
            print(f"  {label:10s}  {prob:.3f}  {bar}")
    except Exception as e:
        print(f"Could not download image: {e}")

    # ------------------------------------------------------------------
    # 3. Image-text alignment — the core CLIP capability
    # ------------------------------------------------------------------
    print("\n=== Image-text alignment ===")
    try:
        sims = encoder.image_text_similarity(img, [
            "a yellow labrador dog",
            "a fluffy golden dog looking at camera",
            "a cat sitting on a fence",
            "a bird flying over water",
        ])
        for text, score in sorted(sims.items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 50)
            print(f"  {score:.3f}  {bar}  {text}")
    except Exception:
        pass

    print("\nDemo complete. Run `python pipeline.py` to build the full search index.")


if __name__ == "__main__":
    main()
