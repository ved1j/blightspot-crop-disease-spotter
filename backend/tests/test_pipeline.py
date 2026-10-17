import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.data import IMG_SIZE, LESION, generate_batch, generate_leaf
from app.inference import count_components, report, severity_label
from app.model import LeafUNet, loss_fn


def test_generator_shapes_and_labels():
    img, mask = generate_leaf(np.random.default_rng(0))
    assert img.shape == (IMG_SIZE, IMG_SIZE, 3) and img.dtype == np.float32
    assert 0 <= img.min() and img.max() <= 1
    assert set(np.unique(mask)) <= {0, 1, 2}
    assert (mask > 0).mean() > 0.1  # a leaf is actually present


def test_lesions_only_on_leaf():
    rng = np.random.default_rng(1)
    for _ in range(20):
        _, mask = generate_leaf(rng, lesion_prob=1.0)
        assert (mask == LESION).any()


def test_model_forward_and_loss():
    x, y = generate_batch(2, np.random.default_rng(2))
    logits = LeafUNet()(torch.from_numpy(x))
    assert logits.shape == (2, 3, IMG_SIZE, IMG_SIZE)
    loss_fn(logits, torch.from_numpy(y)).backward()


def test_count_components():
    m = np.zeros((10, 10), bool)
    m[1:3, 1:3] = True
    m[6:9, 6:9] = True
    m[0, 9] = True  # speck below min size
    assert count_components(m) == 2


@pytest.mark.parametrize("pct,label", [(0, "healthy"), (2, "mild"), (10, "moderate"), (40, "severe")])
def test_severity(pct, label):
    assert severity_label(pct) == label


def test_report_handles_no_leaf():
    assert report(np.zeros((8, 8), np.uint8))["severity"] == "no leaf found"


@pytest.fixture
def client(monkeypatch):
    # Untrained model keeps the tests fast; we only check the API contract here.
    monkeypatch.setattr(main, "load_model", lambda: LeafUNet().eval())
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_analyze_upload(client):
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), (40, 140, 50)).save(buf, "JPEG")
    r = client.post("/analyze", files={"file": ("leaf.jpg", buf.getvalue(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert {"affected_pct", "severity", "lesion_count", "images", "disclaimer"} <= body.keys()
    assert {"original", "overlay", "mask"} <= body["images"].keys()


def test_analyze_rejects_non_image(client):
    r = client.post("/analyze", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_sample_includes_ground_truth(client):
    body = client.get("/sample?seed=3").json()
    assert "ground_truth" in body and "ground_truth" in body["images"]


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "BlightSpot" in r.text
