# 🌿 BlightSpot — Crop Disease Spotter

BlightSpot is a small end-to-end crop-disease demo. It has three parts:

1. **Synthetic leaf data.** Leaves are drawn procedurally, each with lesion patches and a pixel-perfect label mask.
2. **A segmentation model.** A PyTorch U-Net labels every pixel as *background*, *healthy leaf* or *lesion*.
3. **A mobile-friendly upload page.** You take a photo (or pick one) on your phone and see a lesion overlay, the % of leaf area affected, the lesion count and a severity rating.

> **Not a diagnostic tool.** The model has only seen procedurally generated synthetic leaves. It's an educational/portfolio project. Don't use it to make real agronomic decisions.

## How it works

```
 synthetic generator ──► U-Net (3-class) ──► mask ──► report
  leaf + veins + lesions    CE + lesion Dice          % affected, lesion count,
  + ground-truth mask       trained on the fly        severity, overlay PNGs
```

| File | What it does |
|---|---|
| `backend/app/data.py` | Draws each leaf with numpy. It uses fractal-noise backgrounds (soil, concrete, mulch, paper), a pointed leaf outline with a midrib and side veins, and irregular lesions that have a necrotic "target spot" core and a yellow chlorotic halo. Labels: `0` background, `1` leaf, `2` lesion. |
| `backend/app/model.py` | `LeafUNet`, a 3-level U-Net with skip connections (~470k params) that trains on a CPU. The loss is cross-entropy plus soft Dice on the lesion class, so small lesions still count. |
| `backend/app/training.py` | Training loop. Every batch is freshly generated, so there's no dataset on disk. Also has a held-out evaluation (pixel accuracy, lesion Dice). |
| `backend/app/inference.py` | Center-crops and resizes the input, then predicts. It counts lesions (connected components), computes the affected leaf area and severity, and renders the overlay and mask PNGs. |
| `backend/app/main.py` | FastAPI app: `GET /health`, `POST /analyze` (image upload), `GET /sample` (a synthetic leaf plus its ground truth), and `GET /` (the upload page). |
| `frontend/index.html` | A single-file mobile web app with no build step. It has camera capture (`capture="environment"`), a gallery picker, client-side downscaling of big phone photos, dark mode and safe-area insets. |

Severity is based on the share of leaf area covered by lesions: **healthy** < 0.5% ≤ **mild** < 5% ≤ **moderate** < 15% ≤ **severe**.

## Run it

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train.py --steps 1200         # optional: pretrained weights are already committed
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. To try it on your phone, join the same Wi-Fi and open `http://<your-computer-ip>:8000`. Tap **Try a synthetic leaf** to see the prediction next to the ground truth.

If `backend/app/weights/leaf_unet.pt` is missing, the API runs a short training pass on startup, so a fresh checkout still works.

### Tests

```bash
cd backend && pytest -q
```

## API

`POST /analyze`: multipart upload with field `file`.

```json
{
  "leaf_pixels": 6011,
  "lesion_pixels": 412,
  "lesion_count": 4,
  "affected_pct": 6.85,
  "severity": "moderate",
  "images": { "original": "<b64 png>", "overlay": "<b64 png>", "mask": "<b64 png>" },
  "disclaimer": "Demo only: ..."
}
```

`GET /sample?seed=42` returns the same shape. It adds `ground_truth` stats and an `images.ground_truth` mask.

## Limitations / ideas

- Real leaf photos look nothing like these synthetic ones, so expect poor results on them. A natural next step is fine-tuning on a real dataset (e.g. PlantVillage, or a segmentation set such as PlantSeg).
- The model only finds lesions. It doesn't classify the disease type (early blight vs. rust vs. leaf spot).
- It assumes one leaf per image, roughly centered.

## License

MIT
