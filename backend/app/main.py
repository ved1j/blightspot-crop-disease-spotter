"""FastAPI app: segmentation API + serves the mobile web app (PWA) at /."""
from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from .data import generate_leaf
from .inference import (DISCLAIMER, DISEASE_INFO, confidence_map, legend, load_metrics,
                        load_model, mask_to_rgb, overlay, predict, preprocess, report,
                        to_png_b64)

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
SampleKind = Literal["random", "healthy", "early_blight", "rust", "powdery_mildew", "bacterial_spot"]

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["model"] = load_model()
    state["metrics"] = load_metrics()
    yield


app = FastAPI(title="BlightSpot", version="0.2.0", lifespan=lifespan,
              description="Crop-disease lesion segmentation demo (synthetic training data only).")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1024)


def _analyze(img: Image.Image, tta: bool) -> dict:
    t0 = time.perf_counter()
    rgb, x = preprocess(img)
    mask, probs = predict(state["model"], x, tta=tta)
    return {
        **report(mask, probs),
        "images": {
            "original": to_png_b64(rgb),
            "overlay": to_png_b64(overlay(rgb, mask)),
            "mask": to_png_b64(mask_to_rgb(mask)),
            "confidence": to_png_b64(confidence_map(probs)),
        },
        "inference_ms": round(1000 * (time.perf_counter() - t0)),
        "tta": tta,
        "disclaimer": DISCLAIMER,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/classes")
def classes():
    """Class legend (ids + colours) and general information on each disease."""
    return {"classes": legend(), "diseases": DISEASE_INFO}


@app.get("/metrics")
def metrics():
    """Held-out evaluation of the shipped weights (written by train.py)."""
    if not state.get("metrics"):
        raise HTTPException(404, "No metrics.json found; run train.py.")
    return state["metrics"]


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), tta: bool = Query(True, description="test-time augmentation")):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large (max 15 MB).")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)  # respect phone camera orientation
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Could not read that file as an image.")
    return _analyze(img, tta)


@app.get("/sample")
def sample(disease: SampleKind = "random", seed: int | None = None, tta: bool = True):
    """Generate a fresh synthetic leaf, analyse it, and include its ground truth."""
    rng = np.random.default_rng(seed)
    kind = {"random": "random", "healthy": None}.get(disease, disease)
    leaf, truth = generate_leaf(rng, disease=kind)
    result = _analyze(Image.fromarray((leaf * 255).astype(np.uint8)), tta)
    result["images"]["ground_truth"] = to_png_b64(mask_to_rgb(truth))
    gt = report(truth)
    result["ground_truth"] = {k: gt[k] for k in ("diagnosis", "severity", "affected_pct", "lesion_count")}
    return result


# Static PWA last so the API routes above take precedence.
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
