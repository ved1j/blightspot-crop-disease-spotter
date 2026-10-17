"""Train LeafUNet on synthetic leaves and save weights for the API.

    python train.py --steps 1500
"""
import argparse

from app.inference import WEIGHTS_PATH
from app.training import evaluate, train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    model = train(args.steps, args.batch_size, args.lr, save_to=WEIGHTS_PATH)
    print("held-out:", evaluate(model))
