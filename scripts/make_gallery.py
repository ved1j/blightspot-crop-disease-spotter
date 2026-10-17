"""Render docs/gallery.png: synthetic input, ground truth and prediction per disease.

    cd backend && python ../scripts/make_gallery.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.data import generate_leaf  # noqa: E402
from app.inference import load_model, mask_to_rgb, overlay, predict, preprocess  # noqa: E402

KINDS = [("Healthy", None), ("Early blight", "early_blight"), ("Rust", "rust"),
         ("Powdery mildew", "powdery_mildew"), ("Bacterial spot", "bacterial_spot")]
ROWS = ["Input", "Ground truth", "Prediction"]
T, PAD, HEAD, SIDE = 200, 8, 34, 120

model = load_model(train_if_missing=False)
rng = np.random.default_rng(2026)
W = SIDE + len(KINDS) * (T + PAD) + PAD
H = HEAD + len(ROWS) * (T + PAD) + PAD
canvas = Image.new("RGB", (W, H), (244, 246, 241))
draw = ImageDraw.Draw(canvas)

for col, (title, kind) in enumerate(KINDS):
    leaf, truth = generate_leaf(rng, disease=kind)
    rgb, x = preprocess(Image.fromarray((leaf * 255).astype(np.uint8)))
    mask, _ = predict(model, x)
    tiles = [rgb, mask_to_rgb(truth), overlay(rgb, mask)]
    cx = SIDE + PAD + col * (T + PAD)
    draw.text((cx + 4, 10), title, fill=(23, 33, 26))
    for row, tile in enumerate(tiles):
        canvas.paste(Image.fromarray(tile).resize((T, T), Image.NEAREST), (cx, HEAD + row * (T + PAD)))

for row, name in enumerate(ROWS):
    draw.text((10, HEAD + row * (T + PAD) + T // 2 - 6), name, fill=(91, 105, 94))

out = Path(__file__).resolve().parents[1] / "docs" / "gallery.png"
canvas.save(out)
print(f"wrote {out}")
