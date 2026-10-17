"""Load weights, run segmentation, and turn masks into a crop-health report."""
from __future__ import annotations

import base64
import io
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import IMG_SIZE, LEAF, LESION
from .model import LeafUNet

WEIGHTS_PATH = Path(__file__).parent / "weights" / "leaf_unet.pt"

DISCLAIMER = (
    "Demo only: this model was trained exclusively on procedurally generated "
    "synthetic leaves. It is not an agronomic diagnostic tool."
)


def load_model(path: Path = WEIGHTS_PATH, train_if_missing: bool = True) -> LeafUNet:
    model = LeafUNet()
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
    elif train_if_missing:
        from .training import train  # lazy: only needed on a fresh checkout
        print(f"[blightspot] no weights at {path}; running a quick training pass…")
        model = train(steps=300, save_to=path)
    model.eval()
    return model


def preprocess(img: Image.Image) -> tuple[np.ndarray, torch.Tensor]:
    """Center-crop to square, resize to the model's size, return (uint8 RGB, tensor)."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    rgb = np.asarray(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    x = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1)[None]
    return rgb, x


@torch.no_grad()
def predict(model: LeafUNet, x: torch.Tensor) -> np.ndarray:
    return model(x).argmax(1)[0].numpy().astype(np.uint8)


def count_components(binary: np.ndarray, min_pixels: int = 4) -> int:
    """4-connected component count, ignoring specks smaller than min_pixels."""
    seen = np.zeros_like(binary, bool)
    h, w = binary.shape
    count = 0
    for sy, sx in np.argwhere(binary):
        if seen[sy, sx]:
            continue
        size, q = 0, deque([(sy, sx)])
        seen[sy, sx] = True
        while q:
            y, x = q.popleft()
            size += 1
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        count += size >= min_pixels
    return count


def severity_label(pct: float) -> str:
    if pct < 0.5:
        return "healthy"
    if pct < 5:
        return "mild"
    if pct < 15:
        return "moderate"
    return "severe"


def report(mask: np.ndarray) -> dict:
    lesion = int((mask == LESION).sum())
    leaf_total = int((mask == LEAF).sum()) + lesion
    pct = 100 * lesion / leaf_total if leaf_total else 0.0
    return {
        "leaf_pixels": leaf_total,
        "lesion_pixels": lesion,
        "lesion_count": count_components(mask == LESION),
        "affected_pct": round(pct, 2),
        "severity": severity_label(pct) if leaf_total else "no leaf found",
    }


def overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.astype(np.float32)
    out[mask == LESION] = 0.35 * out[mask == LESION] + 0.65 * np.array([255, 40, 90])
    # Outline the leaf so the user can see what the model considered "leaf".
    leafish = mask > 0
    edge = leafish & ~(np.roll(leafish, 1, 0) & np.roll(leafish, -1, 0)
                       & np.roll(leafish, 1, 1) & np.roll(leafish, -1, 1))
    out[edge] = [255, 255, 255]
    return out.clip(0, 255).astype(np.uint8)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    palette = np.array([[20, 20, 20], [70, 170, 80], [235, 50, 90]], np.uint8)
    return palette[mask]


def to_png_b64(arr: np.ndarray, upscale: int = 3) -> str:
    img = Image.fromarray(arr)
    if upscale > 1:
        img = img.resize((arr.shape[1] * upscale, arr.shape[0] * upscale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
