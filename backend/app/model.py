"""A compact U-Net for 6-class leaf / disease segmentation in plain PyTorch."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import DISEASES, NUM_CLASSES


def _block(cin: int, cout: int, dropout: float = 0.0) -> nn.Sequential:
    layers = [
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class LeafUNet(nn.Module):
    """4-level encoder/decoder with skip connections (~1.9M params at base=24)."""

    def __init__(self, in_ch: int = 3, num_classes: int = NUM_CLASSES, base: int = 24):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList([_block(in_ch, c[0]), _block(c[0], c[1]), _block(c[1], c[2])])
        self.bottleneck = _block(c[2], c[3], dropout=0.1)
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in (2, 1, 0)])
        self.dec = nn.ModuleList([_block(c[i] * 2, c[i]) for i in (2, 1, 0)])
        self.head = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], 1))
        return self.head(x)  # logits [N, C, H, W]


# Disease pixels are rare; up-weight them in the cross-entropy term.
CLASS_WEIGHTS = torch.tensor([1.0, 1.0] + [3.0] * len(DISEASES))


def disease_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice averaged over the disease classes, computed over the whole batch."""
    probs = logits.softmax(1)
    scores = []
    for c in DISEASES:
        p, t = probs[:, c], (target == c).float()
        inter = (p * t).sum()
        scores.append((2 * inter + eps) / (p.sum() + t.sum() + eps))
    return torch.stack(scores).mean()


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, weight=CLASS_WEIGHTS.to(logits.device))
    return ce + (1 - disease_dice(logits, target))


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
