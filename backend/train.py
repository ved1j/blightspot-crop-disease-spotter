"""Train LeafUNet on synthetic leaves; saves weights + metrics.json for the API.

    python train.py                  # ~3000 steps; uses CUDA / Apple MPS if available
    python train.py --steps 500      # quick run
"""
import argparse
import json

from app.inference import WEIGHTS_PATH
from app.training import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--workers", type=int, default=4, help="data-generation worker processes")
    args = ap.parse_args()

    train(args.steps, args.batch_size, args.lr, save_to=WEIGHTS_PATH, workers=args.workers)
    m = json.loads(WEIGHTS_PATH.with_name("metrics.json").read_text())
    print(json.dumps({k: m[k] for k in ("pixel_acc", "mean_disease_iou", "diagnosis_acc", "iou")}, indent=2))
