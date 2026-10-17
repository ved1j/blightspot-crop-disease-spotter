"""A small 3-class U-Net (background / leaf / lesion) in plain PyTorch."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import NUM_CLASSES, LESION


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class LeafUNet(nn.Module):
    def __init__(self, in_ch: int = 3, num_classes: int = NUM_CLASSES, base: int = 16):
        super().__init__()
        self.enc1 = _block(in_ch, base)
        self.enc2 = _block(base, base * 2)
        self.enc3 = _block(base * 2, base * 4)
        self.bottleneck = _block(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _block(base * 2, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)  # logits [N, C, H, W]


def lesion_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice on the lesion channel — lesions are small, so CE alone under-weights them."""
    p = logits.softmax(1)[:, LESION]
    t = (target == LESION).float()
    inter = (p * t).sum((1, 2))
    return ((2 * inter + eps) / (p.sum((1, 2)) + t.sum((1, 2)) + eps)).mean()


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target) + (1 - lesion_dice(logits, target))
