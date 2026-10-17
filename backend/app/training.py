"""Training loop on freshly generated synthetic batches (no dataset on disk)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .data import generate_batch
from .model import LeafUNet, lesion_dice, loss_fn


def train(steps: int = 1500, batch_size: int = 16, lr: float = 2e-3, seed: int = 0,
          save_to: Path | None = None, log_every: int = 50) -> LeafUNet:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LeafUNet()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)

    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        x, y = generate_batch(batch_size, rng)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                dice = lesion_dice(logits, y).item()
            print(f"step {step:5d}/{steps}  loss {loss.item():.4f}  lesion-dice {dice:.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    model.eval()
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_to)
        print(f"saved weights to {save_to}")
    return model


@torch.no_grad()
def evaluate(model: LeafUNet, n: int = 64, seed: int = 1234) -> dict:
    """Pixel accuracy and hard lesion Dice on a held-out synthetic batch."""
    x, y = generate_batch(n, np.random.default_rng(seed))
    pred = model(torch.from_numpy(x)).argmax(1).numpy()
    lesion_p, lesion_t = pred == 2, y == 2
    inter = (lesion_p & lesion_t).sum()
    return {
        "pixel_acc": float((pred == y).mean()),
        "lesion_dice": float(2 * inter / max(lesion_p.sum() + lesion_t.sum(), 1)),
    }
