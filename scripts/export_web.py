"""Export everything the browser needs to run BlightSpot with no backend.

Writes into frontend/:
  model/leaf_unet.onnx   the trained U-Net (dynamic batch / height / width)
  model/meta.json        classes, colours, disease info, metrics — shared with the server
  samples/               a pack of synthetic leaves + ground-truth masks for the static demo

    cd backend && python ../scripts/export_web.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.data import CLASSES, DISEASES, IMG_SIZE, generate_leaf  # noqa: E402
from app.inference import (DISCLAIMER, DISEASE_INFO, HEALTHY_INFO, PALETTE,  # noqa: E402
                           load_metrics, load_model, mask_to_rgb, report)

OUT = ROOT / "frontend"
SAMPLES_PER_KIND = 6


def export_onnx(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"},
                      "logits": {0: "batch", 2: "height", 3: "width"}},
        opset_version=17, dynamo=False,
    )


def verify_onnx(model: torch.nn.Module, path: Path) -> float:
    """Max |torch - onnxruntime| on a batch of real samples at two resolutions."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for size in (IMG_SIZE, 128):
        x = torch.rand(2, 3, size, size)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"image": x.numpy()})[0]
        worst = max(worst, float(np.abs(ref - got).max()))
    return worst


def write_meta(path: Path) -> None:
    metrics = load_metrics() or {}
    meta = {
        "img_size": IMG_SIZE,
        "classes": CLASSES,
        "disease_ids": [CLASSES[c] for c in DISEASES],
        "palette": PALETTE.tolist(),
        "names": {c: DISEASE_INFO.get(c, {}).get("name", c.replace("_", " ").title()) for c in CLASSES},
        "disease_info": DISEASE_INFO,
        "healthy_info": HEALTHY_INFO,
        "disclaimer": DISCLAIMER,
        "metrics": {k: metrics.get(k) for k in
                    ("pixel_acc", "mean_disease_iou", "diagnosis_acc", "iou", "n_images", "steps", "device")},
    }
    path.write_text(json.dumps(meta, indent=1))


def write_samples(folder: Path) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.png"):
        old.unlink()
    rng = np.random.default_rng(424242)
    index = []
    kinds = [("healthy", None)] + [(CLASSES[c], CLASSES[c]) for c in DISEASES]
    for kind, disease in kinds:
        for i in range(SAMPLES_PER_KIND):
            img, truth = generate_leaf(rng, disease=disease)
            name = f"{kind}-{i}"
            Image.fromarray((img * 255).astype(np.uint8)).save(folder / f"{name}.png", optimize=True)
            Image.fromarray(mask_to_rgb(truth)).save(folder / f"{name}.mask.png", optimize=True)
            gt = report(truth)
            index.append({"kind": kind, "image": f"samples/{name}.png", "mask": f"samples/{name}.mask.png",
                          "ground_truth": {k: gt[k] for k in ("diagnosis", "severity", "affected_pct", "lesion_count")}})
    (folder / "index.json").write_text(json.dumps(index, indent=1))
    return len(index)


if __name__ == "__main__":
    model = load_model(train_if_missing=False).eval()
    onnx_path = OUT / "model" / "leaf_unet.onnx"
    export_onnx(model, onnx_path)
    err = verify_onnx(model, onnx_path)
    print(f"onnx: {onnx_path.stat().st_size / 1e6:.1f} MB, max |torch - ort| = {err:.2e}")
    assert err < 1e-3, "ONNX output diverges from PyTorch"
    write_meta(OUT / "model" / "meta.json")
    print(f"samples: {write_samples(OUT / 'samples')} images")
