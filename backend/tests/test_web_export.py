"""The static web app ships its own copy of the model + metadata; keep it in sync."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from app.data import CLASSES, DISEASES, IMG_SIZE
from app.inference import DISCLAIMER, DISEASE_INFO, PALETTE, load_model

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
STALE = "frontend export is stale: run `python ../scripts/export_web.py` from backend/"


@pytest.fixture(scope="module")
def meta():
    return json.loads((FRONTEND / "model" / "meta.json").read_text())


def test_meta_matches_backend(meta):
    assert meta["classes"] == CLASSES, STALE
    assert meta["disease_ids"] == [CLASSES[c] for c in DISEASES], STALE
    assert meta["palette"] == PALETTE.tolist(), STALE
    assert meta["disease_info"] == DISEASE_INFO, STALE
    assert meta["disclaimer"] == DISCLAIMER, STALE
    assert meta["img_size"] == IMG_SIZE, STALE


def test_sample_pack_complete(meta):
    index = json.loads((FRONTEND / "samples" / "index.json").read_text())
    kinds = {s["kind"] for s in index}
    assert kinds == {"healthy", *meta["disease_ids"]}
    for s in index:
        assert (FRONTEND / s["image"]).exists() and (FRONTEND / s["mask"]).exists()


def test_onnx_matches_pytorch():
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(str(FRONTEND / "model" / "leaf_unet.onnx"), providers=["CPUExecutionProvider"])
    model = load_model(train_if_missing=False)
    for size in (IMG_SIZE, 128):  # live mode runs at 128 px
        x = torch.rand(1, 3, size, size)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"image": x.numpy()})[0]
        assert np.abs(ref - got).max() < 1e-3, STALE


def test_static_pages_assets_exist():
    for rel in ["index.html", "engine.js", "sw.js", "manifest.webmanifest", "model/leaf_unet.onnx"]:
        assert (FRONTEND / rel).exists(), rel
