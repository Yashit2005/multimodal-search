"""FastAPI endpoint tests with mocked encoder + index."""

import base64
import io
import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock


def make_jpeg_b64(w=32, h=32) -> str:
    img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def make_jpeg_bytes(w=32, h=32) -> bytes:
    img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


MOCK_RESULTS = [
    {"rank": 1, "score": 0.92, "image_id": "1", "image_path": "/img/1.jpg", "caption": "a dog"},
    {"rank": 2, "score": 0.88, "image_id": "2", "image_path": "/img/2.jpg", "caption": "a cat"},
]


@pytest.fixture
def client():
    mock_encoder = MagicMock()
    mock_encoder.encode_text.return_value = np.random.randn(1, 512).astype(np.float32)
    mock_encoder.encode_image.return_value = np.random.randn(1, 512).astype(np.float32)
    mock_encoder.zero_shot_classify.return_value = [{"cat": 0.7, "dog": 0.2, "bird": 0.1}]

    mock_index = MagicMock()
    mock_index.__len__ = MagicMock(return_value=5000)
    mock_index.embed_dim = 512
    mock_index.index_type = "flat"
    mock_index.text_search.return_value = MOCK_RESULTS
    mock_index.image_search.return_value = MOCK_RESULTS
    mock_index.hybrid_search.return_value = MOCK_RESULTS

    from src.serving.api import app, state
    state["encoder"] = mock_encoder
    state["index"] = mock_index
    state["loaded_at"] = 0.0

    from fastapi.testclient import TestClient
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["index_size"] == 5000


def test_index_stats(client):
    r = client.get("/index/stats")
    assert r.status_code == 200
    assert r.json()["n_images"] == 5000


def test_text_search(client):
    r = client.post("/search/text", json={"query": "a dog on the beach", "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 2
    assert body["results"][0]["score"] == 0.92


def test_text_search_top_k_validation(client):
    r = client.post("/search/text", json={"query": "test", "top_k": 200})
    assert r.status_code == 422  # exceeds max of 100


def test_image_search(client):
    r = client.post(
        "/search/image",
        files={"file": ("test.jpg", make_jpeg_bytes(), "image/jpeg")},
        data={"top_k": "5"},
    )
    assert r.status_code == 200
    assert "results" in r.json()


def test_hybrid_search(client):
    r = client.post("/search/hybrid", json={
        "query": "a sunset photo",
        "image_b64": make_jpeg_b64(),
        "alpha": 0.7,
        "top_k": 5,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["alpha"] == 0.7
    assert len(body["results"]) > 0


def test_zero_shot_classify(client):
    r = client.post("/classify/zero-shot", json={
        "image_b64": make_jpeg_b64(),
        "labels": ["cat", "dog", "bird"],
    })
    assert r.status_code == 200
    body = r.json()
    assert "top_label" in body
    assert body["top_label"] == "cat"
    assert len(body["predictions"]) == 3


def test_zero_shot_too_many_labels(client):
    r = client.post("/classify/zero-shot", json={
        "image_b64": make_jpeg_b64(),
        "labels": [f"class_{i}" for i in range(101)],
    })
    assert r.status_code == 400


def test_embed_text(client):
    r = client.post("/embed/text", json={"texts": ["hello world", "test query"]})
    assert r.status_code == 200
    body = r.json()
    assert body["shape"] == [1, 512]  # mock returns (1,512)
