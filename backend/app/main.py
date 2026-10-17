"""FastAPI app: segmentation API + serves the mobile upload page at /."""
from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from .data import generate_leaf
from .inference import (DISCLAIMER, load_model, mask_to_rgb, overlay, predict,
                        preprocess, report, to_png_b64)

FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "index.html"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    torch.set_num_threads(max(1, torch.get_num_threads()))
    state["model"] = load_model()
    yield


app = FastAPI(title="BlightSpot", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _analyze(img: Image.Image) -> dict:
    rgb, x = preprocess(img)
    mask = predict(state["model"], x)
    return {
        **report(mask),
        "images": {
            "original": to_png_b64(rgb),
            "overlay": to_png_b64(overlay(rgb, mask)),
            "mask": to_png_b64(mask_to_rgb(mask)),
        },
        "disclaimer": DISCLAIMER,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large (max 10 MB).")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Could not read that file as an image.")
    return _analyze(img)


@app.get("/sample")
def sample(seed: int | None = None):
    """Generate a fresh synthetic leaf and analyze it, with ground truth for comparison."""
    rng = np.random.default_rng(seed)
    leaf, truth = generate_leaf(rng, lesion_prob=0.9)
    result = _analyze(Image.fromarray((leaf * 255).astype(np.uint8)))
    result["images"]["ground_truth"] = to_png_b64(mask_to_rgb(truth))
    result["ground_truth"] = report(truth)
    return result


@app.get("/")
def index():
    return FileResponse(FRONTEND)
