import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.data import (CLASSES, DISEASES, IMG_SIZE, LEAF, LEAF_SHAPES, generate_batch,
                      generate_leaf)
from app.inference import (components, confidence_map, legend, predict, report,
                           severity_label)
from app.model import LeafUNet, loss_fn
from app.training import dominant_class


# ---------------------------------------------------------------- data

def test_generator_shapes_and_labels():
    img, mask = generate_leaf(np.random.default_rng(0))
    assert img.shape == (IMG_SIZE, IMG_SIZE, 3) and img.dtype == np.float32
    assert 0 <= img.min() and img.max() <= 1
    assert set(np.unique(mask)) <= set(range(len(CLASSES)))
    assert (mask >= LEAF).mean() > 0.05  # a leaf is actually present


@pytest.mark.parametrize("disease", DISEASES)
def test_forced_disease_appears_alone(disease):
    rng = np.random.default_rng(disease)
    for _ in range(5):
        _, mask = generate_leaf(rng, disease=disease)
        present = {c for c in DISEASES if (mask == c).any()}
        assert present == {disease}


def test_healthy_has_no_disease_pixels():
    rng = np.random.default_rng(7)
    for _ in range(10):
        _, mask = generate_leaf(rng, disease=None)
        assert mask.max() <= LEAF


def test_disease_by_name():
    _, mask = generate_leaf(np.random.default_rng(3), disease="rust")
    assert (mask == CLASSES.index("rust")).any()


def test_all_leaf_shapes_render():
    import app.data as data
    rng = np.random.default_rng(0)
    for shape in LEAF_SHAPES:
        leaf, *_ = data._leaf_geometry(rng, IMG_SIZE, shape)
        assert 0.03 < leaf.mean() < 0.7, shape


def test_batch_is_contiguous_nchw():
    x, y = generate_batch(3, np.random.default_rng(1))
    assert x.shape == (3, 3, IMG_SIZE, IMG_SIZE) and x.flags["C_CONTIGUOUS"]
    assert y.shape == (3, IMG_SIZE, IMG_SIZE) and y.dtype == np.int64


# ---------------------------------------------------------------- model

def test_model_forward_and_loss():
    x, y = generate_batch(2, np.random.default_rng(2))
    logits = LeafUNet()(torch.from_numpy(x))
    assert logits.shape == (2, len(CLASSES), IMG_SIZE, IMG_SIZE)
    loss_fn(logits, torch.from_numpy(y)).backward()


def test_tta_prediction_shapes_and_probs():
    model = LeafUNet().eval()
    x = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    mask, probs = predict(model, x, tta=True)
    assert mask.shape == (IMG_SIZE, IMG_SIZE)
    assert probs.shape == (len(CLASSES), IMG_SIZE, IMG_SIZE)
    np.testing.assert_allclose(probs.sum(0), 1, atol=1e-4)


# ---------------------------------------------------------------- reporting

def test_components():
    m = np.zeros((10, 10), bool)
    m[1:3, 1:3] = True
    m[6:9, 6:9] = True
    m[0, 9] = True  # speck below min size
    assert components(m) == 2


@pytest.mark.parametrize("pct,label", [(0, "healthy"), (2, "mild"), (10, "moderate"), (40, "severe")])
def test_severity(pct, label):
    assert severity_label(pct) == label


def test_report_picks_dominant_disease():
    mask = np.full((40, 40), LEAF, np.uint8)
    rust, mildew = CLASSES.index("rust"), CLASSES.index("powdery_mildew")
    mask[0:10, 0:10] = mildew   # 100 px
    mask[20:24, 20:24] = rust   # 16 px
    r = report(mask)
    assert r["diagnosis"] == "powdery_mildew"
    assert [d["id"] for d in r["diseases"]] == ["powdery_mildew", "rust"]
    assert r["diagnosis_info"]["name"] == "Powdery mildew"
    assert r["severity"] == "moderate"


def test_report_handles_no_leaf():
    r = report(np.zeros((8, 8), np.uint8))
    assert r["diagnosis"] == "no_leaf" and r["severity"] == "unknown" and r["warnings"]


def test_dominant_class_threshold():
    mask = np.full((100, 100), LEAF, np.uint8)
    mask[0, 0:10] = DISEASES[0]  # 0.1% of leaf: below threshold
    assert dominant_class(mask) == LEAF
    mask[0:10, 0:10] = DISEASES[0]
    assert dominant_class(mask) == DISEASES[0]


def test_confidence_map_and_legend():
    probs = np.full((len(CLASSES), 4, 4), 1 / len(CLASSES), np.float32)
    assert confidence_map(probs).shape == (4, 4, 3)
    assert [c["id"] for c in legend()] == CLASSES


# ---------------------------------------------------------------- API

@pytest.fixture(scope="module")
def client():
    # Untrained model keeps the tests fast; we only check the API contract here.
    main.load_model = lambda: LeafUNet().eval()
    with TestClient(main.app) as c:
        yield c


def _jpeg(size=(300, 200), color=(40, 140, 50)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_analyze_upload(client):
    r = client.post("/analyze", files={"file": ("leaf.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert {"diagnosis", "diagnosis_info", "severity", "affected_pct", "diseases",
            "confidence", "warnings", "images", "disclaimer", "inference_ms"} <= body.keys()
    assert {"original", "overlay", "mask", "confidence"} <= body["images"].keys()


def test_analyze_without_tta(client):
    r = client.post("/analyze?tta=false", files={"file": ("leaf.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200 and r.json()["tta"] is False


def test_analyze_rejects_non_image(client):
    r = client.post("/analyze", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


@pytest.mark.parametrize("kind", ["random", "healthy", "rust", "bacterial_spot"])
def test_sample_kinds(client, kind):
    body = client.get(f"/sample?disease={kind}&seed=3").json()
    assert "ground_truth" in body and "ground_truth" in body["images"]
    if kind not in ("random", "healthy"):
        assert body["ground_truth"]["diagnosis"] == kind
    if kind == "healthy":
        assert body["ground_truth"]["diagnosis"] == "healthy"


def test_sample_rejects_unknown_disease(client):
    assert client.get("/sample?disease=dragon_pox").status_code == 422


def test_classes(client):
    body = client.get("/classes").json()
    assert len(body["classes"]) == len(CLASSES)
    assert set(body["diseases"]) == {CLASSES[c] for c in DISEASES}


def test_static_pwa_served(client):
    assert "BlightSpot" in client.get("/").text
    assert client.get("/manifest.webmanifest").status_code == 200
    assert client.get("/sw.js").status_code == 200
    assert client.get("/icons/icon-192.png").headers["content-type"] == "image/png"
