"""
CLIP encoder — unified image + text embedding space.

Uses OpenCLIP (open-source CLIP) so you don't need an OpenAI API key.
Both image and text queries embed into the same 512-d / 768-d vector space,
making cross-modal similarity meaningful via cosine distance.

Key insight: CLIP is trained with contrastive loss to pull (image, caption)
pairs together and push non-matching pairs apart. After training, the dot
product between any image embedding and text embedding reflects semantic
alignment — regardless of modality.
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import open_clip
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIPEncoder:
    """
    Wraps OpenCLIP for image + text encoding.

    Args:
        model_name:  OpenCLIP model (default: ViT-B/32 — fast, good quality)
        pretrained:  pretrained weights tag
        device:      'cuda', 'mps', or 'cpu'
        normalize:   L2-normalize embeddings (required for cosine similarity via dot product)
    """

    # Model options with tradeoff notes
    AVAILABLE_MODELS = {
        "ViT-B-32":    {"pretrained": "laion2b_s34b_b79k", "dim": 512,  "note": "fast, good quality — recommended"},
        "ViT-B-16":    {"pretrained": "laion2b_s34b_b88k",  "dim": 512,  "note": "slower, better accuracy"},
        "ViT-L-14":    {"pretrained": "laion2b_s32b_b82k",  "dim": 768,  "note": "best quality, slow"},
        "ViT-H-14":    {"pretrained": "laion2b_s32b_b79k",  "dim": 1024, "note": "research quality, very slow"},
    }

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
        normalize: bool = True,
    ):
        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = device
        self.normalize = normalize
        self.model_name = model_name
        self.embed_dim = self.AVAILABLE_MODELS.get(model_name, {}).get("dim", 512)

        logger.info(f"Loading CLIP model {model_name} ({pretrained}) on {device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(device).eval()
        logger.info(f"CLIP ready. Embedding dim: {self.embed_dim}")

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    def encode_image(
        self,
        images: Union[Image.Image, list[Image.Image], str, list[str]],
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Encode PIL Images or file paths to L2-normalised embeddings.

        Returns:
            np.ndarray of shape (N, embed_dim), dtype float32
        """
        if isinstance(images, (str, Path, Image.Image)):
            images = [images]

        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            pil_images = [
                Image.open(img).convert("RGB") if isinstance(img, (str, Path)) else img.convert("RGB")
                for img in batch
            ]
            tensors = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device == "cuda"):
                emb = self.model.encode_image(tensors)
            all_embeddings.append(emb.cpu().float().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        return self._maybe_normalize(embeddings)

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def encode_text(
        self,
        texts: Union[str, list[str]],
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Encode text strings to L2-normalised embeddings.

        Returns:
            np.ndarray of shape (N, embed_dim), dtype float32
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokens = self.tokenizer(batch).to(self.device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device == "cuda"):
                emb = self.model.encode_text(tokens)
            all_embeddings.append(emb.cpu().float().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        return self._maybe_normalize(embeddings)

    # ------------------------------------------------------------------
    # Zero-shot classification
    # ------------------------------------------------------------------

    def zero_shot_classify(
        self,
        images: Union[Image.Image, list[Image.Image]],
        class_labels: list[str],
        prompt_template: str = "a photo of a {}",
    ) -> list[dict]:
        """
        Classify images into provided categories without any fine-tuning.

        How it works:
          1. Embed all class labels as text using the prompt template
          2. Embed the image(s)
          3. Softmax over dot products → probability per class

        Args:
            images:          PIL Image(s) to classify
            class_labels:    list of category names
            prompt_template: wraps each label (CLIP was trained this way)

        Returns:
            list of {label: prob} dicts, one per image
        """
        prompts = [prompt_template.format(label) for label in class_labels]
        text_embs = self.encode_text(prompts)   # (n_classes, D)
        img_embs = self.encode_image(images)     # (n_images, D)

        # Cosine similarity (both already L2-normalised)
        logits = img_embs @ text_embs.T          # (n_images, n_classes)
        logits = logits * 100                    # temperature scaling (CLIP default)
        probs = self._softmax(logits)

        results = []
        for row in probs:
            results.append({label: float(row[i]) for i, label in enumerate(class_labels)})
        return results

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    def image_text_similarity(
        self, image: Image.Image, texts: list[str]
    ) -> dict[str, float]:
        """Return cosine similarity between one image and a list of texts."""
        img_emb = self.encode_image(image)          # (1, D)
        txt_embs = self.encode_text(texts)           # (N, D)
        sims = (img_emb @ txt_embs.T).squeeze(0)     # (N,)
        return {t: float(s) for t, s in zip(texts, sims)}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _maybe_normalize(self, emb: np.ndarray) -> np.ndarray:
        if self.normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.maximum(norms, 1e-8)
        return emb.astype(np.float32)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
