"""Procedural synthetic leaf generator.

Every image is drawn from scratch with numpy: a noisy background, one leaf
shape with veins, and zero or more lesion patches (dark necrotic centre with a
yellow chlorotic halo). Each image comes with a per-pixel label mask:

    0 = background, 1 = healthy leaf tissue, 2 = lesion
"""
from __future__ import annotations

import numpy as np

IMG_SIZE = 128
NUM_CLASSES = 3
BACKGROUND, LEAF, LESION = 0, 1, 2


def _smooth_noise(rng: np.random.Generator, size: int, scale: int) -> np.ndarray:
    """Cheap value noise: upsample a coarse random grid with bilinear interpolation."""
    coarse = rng.random((scale + 1, scale + 1))
    xs = np.linspace(0, scale, size, endpoint=False)
    x0 = xs.astype(int)
    t = xs - x0
    rows = coarse[x0] * (1 - t)[:, None] + coarse[x0 + 1] * t[:, None]
    return rows[:, x0] * (1 - t)[None, :] + rows[:, x0 + 1] * t[None, :]


def _fractal_noise(rng: np.random.Generator, size: int) -> np.ndarray:
    out = np.zeros((size, size))
    for octave, weight in ((4, 0.5), (8, 0.3), (16, 0.2)):
        out += weight * _smooth_noise(rng, size, octave)
    return out


def _leaf_mask(rng: np.random.Generator, size: int) -> tuple[np.ndarray, tuple, float, np.ndarray, np.ndarray]:
    """Return a boolean leaf mask plus the leaf frame (centre, angle, local coords)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    cx = size / 2 + rng.uniform(-8, 8)
    cy = size / 2 + rng.uniform(-8, 8)
    angle = rng.uniform(0, np.pi)
    length = rng.uniform(0.36, 0.46) * size  # half-length along the midrib
    width = length * rng.uniform(0.38, 0.6)  # max half-width

    dx, dy = xx - cx, yy - cy
    u = dx * np.cos(angle) + dy * np.sin(angle)  # along midrib
    v = -dx * np.sin(angle) + dy * np.cos(angle)  # across midrib
    un = u / length  # -1 (base) .. 1 (tip)

    # Leaf outline: widest a bit below the middle, tapering to a pointed tip.
    profile = np.clip(1 - un**2, 0, None) ** 0.75 * (1 - 0.25 * un)
    wobble = 1 + 0.06 * np.sin(un * rng.uniform(8, 14) + rng.uniform(0, 6))
    mask = (np.abs(un) <= 1) & (np.abs(v) <= width * profile * wobble)
    return mask, (cx, cy), angle, u, v


def generate_leaf(rng: np.random.Generator | None = None, size: int = IMG_SIZE,
                  lesion_prob: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Generate one (image, mask) pair.

    image: float32 array (size, size, 3) in [0, 1]
    mask:  uint8 array (size, size) with values in {0, 1, 2}
    """
    rng = rng or np.random.default_rng()

    # Background: soil / bench / table tones with texture.
    bg_base = rng.choice([
        [0.45, 0.33, 0.22],  # soil
        [0.70, 0.66, 0.58],  # concrete
        [0.25, 0.22, 0.20],  # dark mulch
        [0.82, 0.80, 0.76],  # paper
    ]) + rng.normal(0, 0.05, 3)
    tex = _fractal_noise(rng, size)[..., None]
    img = np.clip(bg_base * (0.75 + 0.5 * tex), 0, 1)

    leaf, _, _, u, v = _leaf_mask(rng, size)
    label = np.zeros((size, size), np.uint8)
    label[leaf] = LEAF

    # Leaf colour with mottling and veins.
    green = np.array([rng.uniform(0.15, 0.3), rng.uniform(0.45, 0.65), rng.uniform(0.1, 0.25)])
    leaf_tex = _fractal_noise(rng, size)[..., None]
    leaf_rgb = green * (0.8 + 0.4 * leaf_tex)
    midrib = np.exp(-(v / 1.2) ** 2)
    vein_freq = rng.uniform(0.25, 0.4)
    side_veins = (np.cos((u - np.abs(v) * 1.2) * vein_freq) > 0.93) * np.exp(-np.abs(v) / 25)
    veins = np.clip(midrib + 0.6 * side_veins, 0, 1)[..., None]
    leaf_rgb = leaf_rgb * (1 - 0.35 * veins) + np.array([0.75, 0.85, 0.55]) * 0.35 * veins
    img = np.where(leaf[..., None], leaf_rgb, img)

    # Lesions: only placed on leaf tissue.
    if rng.random() < lesion_prob:
        leaf_pts = np.argwhere(leaf)
        yy, xx = np.mgrid[0:size, 0:size].astype(float)
        for _ in range(rng.integers(1, 7)):
            py, px = leaf_pts[rng.integers(len(leaf_pts))]
            r = rng.uniform(2.5, 9)
            # Irregular blob: radius perturbed by angle-dependent noise.
            theta = np.arctan2(yy - py, xx - px)
            k = rng.integers(3, 7)
            r_theta = r * (1 + 0.3 * np.sin(k * theta + rng.uniform(0, 6)))
            dist = np.hypot(yy - py, xx - px) / r_theta
            core = (dist < 1) & leaf
            halo = (dist >= 1) & (dist < rng.uniform(1.3, 1.7)) & leaf

            necrotic = np.array([rng.uniform(0.3, 0.45), rng.uniform(0.18, 0.28), rng.uniform(0.08, 0.14)])
            ring_tex = 0.85 + 0.3 * np.cos(dist * 9)[..., None]  # concentric "target spot" rings
            img = np.where(core[..., None], np.clip(necrotic * ring_tex, 0, 1), img)
            chlorotic = np.array([0.78, 0.74, 0.25])
            img = np.where(halo[..., None], 0.45 * img + 0.55 * chlorotic, img)
            label[core] = LESION

    # Global lighting + sensor noise.
    img = img * rng.uniform(0.8, 1.15) + rng.normal(0, 0.025, img.shape)
    return np.clip(img, 0, 1).astype(np.float32), label


def generate_batch(n: int, rng: np.random.Generator | None = None, size: int = IMG_SIZE):
    """Return (images[N,3,H,W] float32, masks[N,H,W] int64) ready for PyTorch."""
    rng = rng or np.random.default_rng()
    imgs, masks = zip(*(generate_leaf(rng, size) for _ in range(n)))
    return np.stack(imgs).transpose(0, 3, 1, 2), np.stack(masks).astype(np.int64)
