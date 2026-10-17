"""Training + evaluation on freshly generated synthetic leaves (no dataset on disk)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .data import CLASSES, DISEASES, LEAF, generate_batch
from .model import LeafUNet, loss_fn, pick_device


class SyntheticLeaves(IterableDataset):
    """Endless stream of synthetic batches; each worker gets its own RNG stream."""

    def __init__(self, batch_size: int, seed: int):
        self.batch_size, self.seed = batch_size, seed

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        while True:
            x, y = generate_batch(self.batch_size, rng)
            yield torch.from_numpy(x), torch.from_numpy(y)


def dominant_class(mask: np.ndarray, min_pct: float = 0.5) -> int:
    """Image-level label: the disease covering the most leaf area, or LEAF if healthy."""
    leaf_area = max(int((mask >= LEAF).sum()), 1)
    areas = {c: int((mask == c).sum()) for c in DISEASES}
    best = max(areas, key=areas.get)
    return best if 100 * areas[best] / leaf_area >= min_pct else LEAF


@torch.no_grad()
def evaluate(model: LeafUNet, n: int = 256, seed: int = 10_000, batch: int = 32) -> dict:
    """Per-class IoU, pixel accuracy and image-level diagnosis accuracy on a held-out set."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    inter = np.zeros(len(CLASSES))
    union = np.zeros(len(CLASSES))
    correct_px = total_px = correct_dx = 0
    confusion = np.zeros((len(CLASSES), len(CLASSES)), int)  # image-level: rows=true, cols=pred
    for start in range(0, n, batch):
        x, y = generate_batch(min(batch, n - start), rng)
        pred = model(torch.from_numpy(x).to(device)).argmax(1).cpu().numpy()
        correct_px += (pred == y).sum()
        total_px += y.size
        for c in range(len(CLASSES)):
            inter[c] += ((pred == c) & (y == c)).sum()
            union[c] += ((pred == c) | (y == c)).sum()
        for p, t in zip(pred, y):
            tc, pc = dominant_class(t), dominant_class(p)
            confusion[tc, pc] += 1
            correct_dx += tc == pc
    model.train(was_training)
    iou = {CLASSES[c]: float(inter[c] / union[c]) if union[c] else None for c in range(len(CLASSES))}
    disease_ious = [iou[CLASSES[c]] for c in DISEASES if iou[CLASSES[c]] is not None]
    return {
        "pixel_acc": float(correct_px / total_px),
        "iou": iou,
        "mean_disease_iou": float(np.mean(disease_ious)),
        "diagnosis_acc": float(correct_dx / n),
        "diagnosis_confusion": {
            "labels": ["healthy" if c == LEAF else CLASSES[c] for c in [LEAF, *DISEASES]],
            "matrix": confusion[np.ix_([LEAF, *DISEASES], [LEAF, *DISEASES])].tolist(),
        },
        "n_images": n,
    }


def train(steps: int = 3000, batch_size: int = 24, lr: float = 3e-3, seed: int = 0,
          save_to: Path | None = None, eval_every: int = 500, workers: int = 0,
          log_every: int = 100) -> LeafUNet:
    torch.manual_seed(seed)
    device = pick_device()
    model = LeafUNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    loader = DataLoader(SyntheticLeaves(batch_size, seed), batch_size=None, num_workers=workers,
                        persistent_workers=workers > 0)
    print(f"training on {device} | {sum(p.numel() for p in model.parameters()):,} params | "
          f"{steps} steps x {batch_size} images", flush=True)

    best, best_state, history = -1.0, None, []
    t0 = time.time()
    model.train()
    for step, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0:
            print(f"step {step:5d}/{steps}  loss {loss.item():.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if step % eval_every == 0 or step == steps:
            m = evaluate(model, n=96)
            history.append({"step": step, "loss": float(loss.item()), **{k: m[k] for k in
                            ("pixel_acc", "mean_disease_iou", "diagnosis_acc")}})
            print(f"  val  mIoU(disease) {m['mean_disease_iou']:.3f}  dx-acc {m['diagnosis_acc']:.3f}",
                  flush=True)
            if m["mean_disease_iou"] > best:
                best = m["mean_disease_iou"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if step >= steps:
            break

    model.load_state_dict(best_state)
    model.eval()
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_to)
        metrics = {**evaluate(model), "steps": steps, "batch_size": batch_size,
                   "train_seconds": round(time.time() - t0), "device": str(device), "history": history}
        save_to.with_name("metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"saved weights + metrics to {save_to.parent}")
    return model.cpu()
