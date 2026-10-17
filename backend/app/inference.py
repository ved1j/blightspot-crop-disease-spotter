"""Load weights, run segmentation, and turn masks into a crop-health report."""
from __future__ import annotations

import base64
import io
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import BACKGROUND, CLASSES, DISEASES, IMG_SIZE, LEAF
from .model import LeafUNet

WEIGHTS_PATH = Path(__file__).parent / "weights" / "leaf_unet.pt"
METRICS_PATH = WEIGHTS_PATH.with_name("metrics.json")

DISCLAIMER = (
    "Demo only: this model was trained exclusively on procedurally generated "
    "synthetic leaves. It is not an agronomic diagnostic tool; confirm any real "
    "diagnosis with a local extension service or plant pathologist."
)

# Overlay / mask colours (RGB) per class.
PALETTE = np.array([
    [20, 20, 20],     # background
    [70, 170, 80],    # leaf
    [150, 82, 30],    # early blight
    [255, 140, 20],   # rust
    [215, 225, 255],  # powdery mildew
    [205, 40, 200],   # bacterial spot
], np.uint8)

# General, textbook-level information shown alongside a result. Not a treatment plan.
DISEASE_INFO = {
    "early_blight": {
        "name": "Early blight",
        "cause": "Fungus (Alternaria spp.)",
        "symptoms": "Brown lesions with concentric 'target' rings and a yellow halo, "
                    "usually starting on older, lower leaves.",
        "tips": [
            "Remove and dispose of infected lower leaves.",
            "Water at the base; avoid wetting foliage.",
            "Mulch to stop soil splashing onto leaves.",
            "Rotate crops away from tomato/potato family for 2-3 years.",
        ],
    },
    "rust": {
        "name": "Rust",
        "cause": "Fungi (order Pucciniales)",
        "symptoms": "Clusters of small, raised orange-to-brown pustules that rub off as powdery spores, "
                    "often on leaf undersides.",
        "tips": [
            "Pick off and bag affected leaves early.",
            "Improve air circulation by spacing and pruning.",
            "Avoid overhead watering, especially in the evening.",
            "Prefer rust-resistant varieties next season.",
        ],
    },
    "powdery_mildew": {
        "name": "Powdery mildew",
        "cause": "Fungi (order Erysiphales)",
        "symptoms": "White to grey powdery patches on the leaf surface that can spread to cover the leaf.",
        "tips": [
            "Increase sunlight and airflow around plants.",
            "Remove heavily coated leaves.",
            "Avoid excess nitrogen fertiliser, which favours soft growth.",
            "Sulfur or potassium-bicarbonate products are common organic options.",
        ],
    },
    "bacterial_spot": {
        "name": "Bacterial spot",
        "cause": "Bacteria (Xanthomonas spp.)",
        "symptoms": "Many small, dark, angular, water-soaked spots, sometimes with thin yellow halos.",
        "tips": [
            "Don't handle plants while they are wet.",
            "Water at soil level; splashing spreads the bacteria.",
            "Sanitise tools and remove badly infected plants.",
            "Use certified disease-free seed and rotate crops.",
        ],
    },
}
HEALTHY_INFO = {
    "name": "Healthy",
    "cause": None,
    "symptoms": "No disease symptoms detected on the visible leaf area.",
    "tips": ["Keep monitoring weekly; early detection makes management much easier."],
}


def load_model(path: Path = WEIGHTS_PATH, train_if_missing: bool = True) -> LeafUNet:
    model = LeafUNet()
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
    elif train_if_missing:
        from .training import train  # lazy: only needed on a fresh checkout
        print(f"[blightspot] no weights at {path}; running a quick training pass…")
        model = train(steps=400, save_to=path)
    model.eval()
    return model


def load_metrics() -> dict | None:
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None


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
def predict(model: LeafUNet, x: torch.Tensor, tta: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask[H,W] uint8, probs[C,H,W] float32).

    With tta=True, softmax outputs are averaged over the identity, horizontal flip,
    vertical flip and 90-degree rotation (leaves have no canonical orientation).
    """
    views = [(lambda t: t, lambda t: t)]
    if tta:
        views += [
            (lambda t: t.flip(-1), lambda t: t.flip(-1)),
            (lambda t: t.flip(-2), lambda t: t.flip(-2)),
            (lambda t: t.rot90(1, (-2, -1)), lambda t: t.rot90(-1, (-2, -1))),
        ]
    batch = torch.cat([fwd(x) for fwd, _ in views])
    out = model(batch).softmax(1)
    probs = torch.stack([inv(out[i:i + 1]) for i, (_, inv) in enumerate(views)]).mean(0)[0]
    probs = probs.numpy()
    return probs.argmax(0).astype(np.uint8), probs


def components(binary: np.ndarray, min_pixels: int = 3) -> int:
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


def report(mask: np.ndarray, probs: np.ndarray | None = None) -> dict:
    """Summarise a class mask (and optional probabilities) into a diagnosis."""
    leaf_total = int((mask >= LEAF).sum())
    leaf_frac = leaf_total / mask.size
    diseases = []
    for c in DISEASES:
        px = int((mask == c).sum())
        if not px:
            continue
        pct = 100 * px / max(leaf_total, 1)
        entry = {
            "id": CLASSES[c],
            "name": DISEASE_INFO[CLASSES[c]]["name"],
            "pixels": px,
            "affected_pct": round(pct, 2),
            "lesion_count": components(mask == c),
        }
        if probs is not None:
            entry["confidence"] = round(float(probs[c][mask == c].mean()), 3)
        diseases.append(entry)
    diseases.sort(key=lambda d: d["pixels"], reverse=True)
    affected = sum(d["pixels"] for d in diseases)
    total_pct = 100 * affected / leaf_total if leaf_total else 0.0

    significant = [d for d in diseases if d["affected_pct"] >= 0.5]
    primary = significant[0]["id"] if significant else "healthy"

    warnings = []
    if leaf_frac < 0.04:
        warnings.append("Very little leaf detected. Fill the frame with a single leaf.")
    confidence = None
    if probs is not None and leaf_total:
        confidence = round(float(probs.max(0)[mask >= LEAF].mean()), 3)
        if confidence < 0.8:
            warnings.append("Low model confidence: this photo looks unlike the synthetic training data.")

    return {
        "diagnosis": primary if leaf_total and leaf_frac >= 0.01 else "no_leaf",
        "diagnosis_info": DISEASE_INFO.get(primary, HEALTHY_INFO),
        "severity": severity_label(total_pct) if leaf_total else "unknown",
        "affected_pct": round(total_pct, 2),
        "leaf_pixels": leaf_total,
        "leaf_coverage_pct": round(100 * leaf_frac, 1),
        "lesion_pixels": affected,
        "lesion_count": sum(d["lesion_count"] for d in diseases),
        "diseases": diseases,
        "confidence": confidence,
        "warnings": warnings,
    }


# ---------------------------------------------------------------- rendering

def overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.astype(np.float32)
    for c in DISEASES:
        m = mask == c
        out[m] = 0.3 * out[m] + 0.7 * PALETTE[c]
    leafish = mask != BACKGROUND
    edge = leafish & ~(np.roll(leafish, 1, 0) & np.roll(leafish, -1, 0)
                       & np.roll(leafish, 1, 1) & np.roll(leafish, -1, 1))
    out[edge] = [255, 255, 255]
    return out.clip(0, 255).astype(np.uint8)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return PALETTE[mask]


def confidence_map(probs: np.ndarray) -> np.ndarray:
    """Max-probability heatmap: dark purple = unsure, yellow = confident (viridis-like)."""
    stops = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], float)
    conf = (probs.max(0) - 1 / probs.shape[0]) / (1 - 1 / probs.shape[0])
    t = np.clip(conf, 0, 1) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    f = (t - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def to_png_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def legend() -> list[dict]:
    return [{"id": c, "name": DISEASE_INFO.get(c, {}).get("name", c.replace("_", " ").title()),
             "color": "#%02x%02x%02x" % tuple(PALETTE[i])} for i, c in enumerate(CLASSES)]
