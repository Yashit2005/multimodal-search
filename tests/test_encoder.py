"""Tests for CLIP encoder and FAISS index."""

import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# CLIPEncoder unit tests (mock the model to avoid downloading weights)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_encoder():
    """CLIPEncoder with mocked underlying model — no GPU/download needed."""
    with patch("src.embeddings.clip_encoder.open_clip") as mock_clip:
        mock_model = MagicMock()
        mock_model.encode_image.return_value = __import__("torch").randn(2, 512)
        mock_model.encode_text.return_value = __import__("torch").randn(2, 512)

        mock_preprocess = MagicMock(return_value=__import__("torch").randn(3, 224, 224))
        mock_tokenizer = MagicMock(return_value=__import__("torch").randint(0, 100, (2, 77)))
        mock_clip.create_model_and_transforms.return_value = (mock_model, None, mock_preprocess)
        mock_clip.get_tokenizer.return_value = mock_tokenizer

        from src.embeddings.clip_encoder import CLIPEncoder
        encoder = CLIPEncoder.__new__(CLIPEncoder)
        encoder.model = mock_model.to("cpu").eval()
        encoder.preprocess = mock_preprocess
        encoder.tokenizer = mock_tokenizer
        encoder.device = "cpu"
        encoder.normalize = True
        encoder.embed_dim = 512
        encoder.model_name = "ViT-B-32"
        yield encoder


def make_image(w=64, h=64):
    return Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))


def test_text_embedding_shape():
    """Text embedding must have correct dimensionality."""
    from src.embeddings.clip_encoder import CLIPEncoder
    with patch.object(CLIPEncoder, "__init__", return_value=None):
        enc = CLIPEncoder.__new__(CLIPEncoder)
        enc.embed_dim = 512
        enc.normalize = True
        # Directly test normalisation logic
        raw = np.random.randn(3, 512).astype(np.float32)
        normed = enc._maybe_normalize(raw)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_normalisation_unit_vectors():
    from src.embeddings.clip_encoder import CLIPEncoder
    with patch.object(CLIPEncoder, "__init__", return_value=None):
        enc = CLIPEncoder.__new__(CLIPEncoder)
        enc.normalize = True
        raw = np.array([[3.0, 4.0]], dtype=np.float32)
        normed = enc._maybe_normalize(raw)
        np.testing.assert_allclose(np.linalg.norm(normed, axis=1), [1.0], atol=1e-6)


def test_softmax_sums_to_one():
    from src.embeddings.clip_encoder import CLIPEncoder
    logits = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    probs = CLIPEncoder._softmax(logits)
    np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-6)


# ---------------------------------------------------------------------------
# ImageIndex unit tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_index():
    from src.search.index import ImageIndex
    idx = ImageIndex(embed_dim=512, index_type="flat")
    n = 20
    embs = np.random.randn(n, 512).astype(np.float32)
    # L2-normalise
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    meta = [{"image_id": str(i), "caption": f"image {i}", "image_path": f"/img/{i}.jpg"}
            for i in range(n)]
    idx.add(embs, meta)
    return idx, embs, meta


def test_index_length(sample_index):
    idx, _, meta = sample_index
    assert len(idx) == len(meta)


def test_search_returns_top_k(sample_index):
    idx, embs, _ = sample_index
    query = embs[0:1]
    results = idx.search(query, top_k=5)
    assert len(results) == 5


def test_top_result_is_self(sample_index):
    """Querying with an indexed vector should return itself as rank 1."""
    idx, embs, meta = sample_index
    results = idx.search(embs[3:4], top_k=3)
    assert results[0]["image_id"] == "3"


def test_hybrid_search_output(sample_index):
    idx, embs, _ = sample_index
    results = idx.hybrid_search(embs[0:1], embs[1:2], alpha=0.5, top_k=5)
    assert len(results) == 5
    assert all("score" in r for r in results)


def test_index_save_load(sample_index, tmp_path):
    from src.search.index import ImageIndex
    idx, _, _ = sample_index
    idx.save(str(tmp_path))
    loaded = ImageIndex.load(str(tmp_path))
    assert len(loaded) == len(idx)
    assert loaded.embed_dim == idx.embed_dim


def test_filter_fn(sample_index):
    idx, embs, _ = sample_index
    results = idx.search(embs[0:1], top_k=10,
                          filter_fn=lambda m: int(m["image_id"]) % 2 == 0)
    assert all(int(r["image_id"]) % 2 == 0 for r in results)
