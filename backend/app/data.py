"""Procedural synthetic leaf generator.

Every image is drawn from scratch with numpy: a textured background, a leaf of
one of several shapes, zero or more disease symptoms, and photo-style
degradations (shadows, blur, colour casts, noise). Each image comes with a
per-pixel label mask whose values are the indices of ``CLASSES``.

The four diseases are drawn to mimic their textbook visual signatures:

* early blight     - large brown lesions with concentric "target" rings and a yellow halo
* rust             - clusters of tiny raised orange-brown pustules
* powdery mildew   - diffuse, irregular whitish-grey patches on the surface
* bacterial spot   - many small, dark, angular spots, sometimes with thin halos
"""
from __future__ import annotations

import numpy as np

IMG_SIZE = 160

CLASSES = ["background", "leaf", "early_blight", "rust", "powdery_mildew", "bacterial_spot"]
NUM_CLASSES = len(CLASSES)
BACKGROUND, LEAF, EARLY_BLIGHT, RUST, POWDERY_MILDEW, BACTERIAL_SPOT = range(NUM_CLASSES)
DISEASES = [EARLY_BLIGHT, RUST, POWDERY_MILDEW, BACTERIAL_SPOT]
DISEASE_NAMES = {c: CLASSES[c] for c in DISEASES}
LEAF_SHAPES = ["ovate", "lanceolate", "cordate", "lobed"]


# ---------------------------------------------------------------- noise utils

def _smooth_noise(rng: np.random.Generator, size: int, scale: int) -> np.ndarray:
    """Value noise: bilinear upsampling of a coarse random grid."""
    coarse = rng.random((scale + 1, scale + 1))
    xs = np.linspace(0, scale, size, endpoint=False)
    x0 = xs.astype(int)
    t = xs - x0
    t = t * t * (3 - 2 * t)  # smoothstep, avoids visible grid creases
    rows = coarse[x0] * (1 - t)[:, None] + coarse[x0 + 1] * t[:, None]
    return rows[:, x0] * (1 - t)[None, :] + rows[:, x0 + 1] * t[None, :]


def fractal_noise(rng: np.random.Generator, size: int,
                  octaves=((4, 0.5), (8, 0.3), (16, 0.2))) -> np.ndarray:
    out = np.zeros((size, size))
    for scale, weight in octaves:
        out += weight * _smooth_noise(rng, size, scale)
    return out


def _box_blur(img: np.ndarray, r: int) -> np.ndarray:
    """Separable box blur on an (H, W, C) array using cumulative sums."""
    if r <= 0:
        return img
    k = 2 * r + 1
    for axis in (0, 1):
        pad = [(0, 0)] * img.ndim
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(img, pad, mode="edge"), axis=axis)
        img = (np.take(c, range(k, c.shape[axis]), axis=axis)
               - np.take(c, range(0, c.shape[axis] - k), axis=axis)) / k
    return img


def _window(cy: float, cx: float, r: float, size: int):
    """Bounding-box slices around a point, plus local coordinate grids."""
    y0, y1 = max(int(cy - r - 1), 0), min(int(cy + r + 2), size)
    x0, x1 = max(int(cx - r - 1), 0), min(int(cx + r + 2), size)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)
    return (slice(y0, y1), slice(x0, x1)), yy - cy, xx - cx


# ---------------------------------------------------------------- leaf shapes

def _leaf_geometry(rng: np.random.Generator, size: int, shape: str, offset=(0.0, 0.0)):
    """Return (leaf mask, u, v, petiole mask) in the leaf's local frame.

    u runs along the midrib (base -> tip), v across it.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    cx = size / 2 + (rng.uniform(-0.08, 0.08) + offset[1]) * size
    cy = size / 2 + (rng.uniform(-0.08, 0.08) + offset[0]) * size
    angle = rng.uniform(0, 2 * np.pi)
    length = rng.uniform(0.34, 0.45) * size  # half-length along midrib

    dx, dy = xx - cx, yy - cy
    u = dx * np.cos(angle) + dy * np.sin(angle)
    v = -dx * np.sin(angle) + dy * np.cos(angle)
    un = u / length

    if shape == "lobed":
        # Palmate outline in polar coordinates, e.g. maple / grape / cucurbit leaves.
        lobes = rng.choice([3, 5])
        r = np.hypot(u, v)
        theta = np.arctan2(v, u)
        depth = rng.uniform(0.18, 0.32)
        radius = length * 0.85 * (1 - depth + depth * np.abs(np.cos(lobes * theta / 2)) ** 0.6)
        radius *= 1 - 0.35 * np.clip(-np.cos(theta), 0, None) ** 4  # notch at the base
        mask = r <= radius
    else:
        if shape == "lanceolate":
            width = length * rng.uniform(0.2, 0.3)
            profile = np.clip(1 - un**2, 0, None) ** 0.9
        elif shape == "cordate":
            width = length * rng.uniform(0.6, 0.8)
            profile = np.clip(1 - un**2, 0, None) ** 0.6 * (1 - 0.35 * un)
            profile *= np.where(un < -0.7, np.clip((np.abs(v) / (width + 1e-6)) * 3, 0, 1), 1)
        else:  # ovate
            width = length * rng.uniform(0.38, 0.6)
            profile = np.clip(1 - un**2, 0, None) ** 0.75 * (1 - 0.25 * un)
        wobble = 1 + 0.05 * np.sin(un * rng.uniform(8, 14) + rng.uniform(0, 6))
        if rng.random() < 0.5:  # serrated margin
            wobble *= 1 + 0.035 * np.abs(np.sin(un * rng.uniform(40, 70)))
        mask = (np.abs(un) <= 1) & (np.abs(v) <= width * profile * wobble)

    stem_len = length * rng.uniform(0.15, 0.35)
    petiole = (u < 0) & (u > -length - stem_len) & (np.abs(v) < rng.uniform(1.0, 1.8)) & ~mask
    return mask, u, v, petiole


# ---------------------------------------------------------------- symptoms

def _draw_early_blight(rng, img, label, leaf, size):
    pts = np.argwhere(leaf)
    for _ in range(rng.integers(1, 6)):
        py, px = pts[rng.integers(len(pts))]
        r = rng.uniform(3, 11)
        sl, ly, lx = _window(py, px, r * 1.8, size)
        theta = np.arctan2(ly, lx)
        k = rng.integers(3, 7)
        r_theta = r * (1 + 0.25 * np.sin(k * theta + rng.uniform(0, 6)))
        dist = np.hypot(ly, lx) / r_theta
        on_leaf = leaf[sl]
        core = (dist < 1) & on_leaf
        halo = (dist >= 1) & (dist < rng.uniform(1.3, 1.7)) & on_leaf

        brown = np.array([rng.uniform(0.32, 0.45), rng.uniform(0.2, 0.28), rng.uniform(0.08, 0.13)])
        rings = (0.78 + 0.35 * (np.cos(dist * rng.uniform(8, 12)) > 0.2))[..., None]
        patch = img[sl]
        patch[core] = np.clip(brown * rings, 0, 1)[core]
        patch[halo] = 0.45 * patch[halo] + 0.55 * np.array([0.8, 0.75, 0.22])
        label[sl][core] = EARLY_BLIGHT


def _draw_rust(rng, img, label, leaf, size):
    pts = np.argwhere(leaf)
    for _ in range(rng.integers(1, 4)):
        cy, cx = pts[rng.integers(len(pts))]
        spread = rng.uniform(6, 16)
        # Faint yellowing under the cluster (not labelled as disease).
        sl, ly, lx = _window(cy, cx, spread * 1.5, size)
        fade = np.exp(-(ly**2 + lx**2) / (2 * spread**2))[..., None] * leaf[sl][..., None]
        img[sl] = img[sl] * (1 - 0.35 * fade) + np.array([0.75, 0.7, 0.2]) * 0.35 * fade
        for _ in range(rng.integers(8, 30)):
            py, px = cy + rng.normal(0, spread / 2), cx + rng.normal(0, spread / 2)
            if not (0 <= py < size and 0 <= px < size) or not leaf[int(py), int(px)]:
                continue
            r = rng.uniform(1.0, 2.2)
            sl, ly, lx = _window(py, px, r, size)
            d = np.hypot(ly, lx) / r
            dot = (d < 1) & leaf[sl]
            orange = np.array([rng.uniform(0.75, 0.9), rng.uniform(0.38, 0.5), rng.uniform(0.05, 0.12)])
            shade = (1.15 - 0.35 * d)[..., None]  # raised, lit from the centre
            img[sl][dot] = np.clip(orange * shade, 0, 1)[dot]
            label[sl][dot] = RUST


def _draw_powdery_mildew(rng, img, label, leaf, size):
    pts = np.argwhere(leaf)
    fuzz = fractal_noise(rng, size, ((8, 0.35), (16, 0.35), (32, 0.3)))
    for _ in range(rng.integers(2, 7)):
        py, px = pts[rng.integers(len(pts))]
        r = rng.uniform(5, 16)
        sl, ly, lx = _window(py, px, r * 1.6, size)
        theta = np.arctan2(ly, lx)
        warp = 1 + 0.35 * np.sin(rng.integers(2, 5) * theta + rng.uniform(0, 6))
        falloff = np.clip(1 - np.hypot(ly, lx) / (r * warp), 0, 1)
        density = falloff ** 0.6 * (0.3 + 1.4 * fuzz[sl]) - 0.35
        covered = (density > 0.15) & leaf[sl]
        alpha = np.clip(density * 1.6, 0, 0.85)[..., None]
        # Powdery texture: speckled, not a flat white disc.
        speckle = rng.uniform(0.85, 1.05, density.shape)[..., None]
        white = np.array([0.9, 0.91, 0.87]) * speckle
        patch = img[sl]
        blended = patch * (1 - alpha) + white * alpha
        patch[leaf[sl]] = blended[leaf[sl]]
        label[sl][covered] = POWDERY_MILDEW


def _draw_bacterial_spot(rng, img, label, leaf, size):
    pts = np.argwhere(leaf)
    halo_color = np.array([0.72, 0.72, 0.25])
    for _ in range(rng.integers(10, 45)):
        py, px = pts[rng.integers(len(pts))]
        r = rng.uniform(1.0, 2.8)
        sl, ly, lx = _window(py, px, r * 2, size)
        theta = np.arctan2(ly, lx)
        # Angular, vein-limited spots: low-order polygon-ish outline.
        r_theta = r * (1 + 0.3 * np.cos(rng.integers(3, 6) * theta + rng.uniform(0, 6)))
        d = np.hypot(ly, lx) / r_theta
        spot = (d < 1) & leaf[sl]
        patch = img[sl]
        if rng.random() < 0.6:
            halo = (d >= 1) & (d < 1.5) & leaf[sl]
            patch[halo] = 0.5 * patch[halo] + 0.5 * halo_color
        dark = np.array([rng.uniform(0.08, 0.16), rng.uniform(0.07, 0.12), rng.uniform(0.04, 0.08)])
        patch[spot] = dark
        label[sl][spot] = BACTERIAL_SPOT


DRAWERS = {
    EARLY_BLIGHT: _draw_early_blight,
    RUST: _draw_rust,
    POWDERY_MILDEW: _draw_powdery_mildew,
    BACTERIAL_SPOT: _draw_bacterial_spot,
}


# ---------------------------------------------------------------- composition

BACKGROUNDS = [
    [0.45, 0.33, 0.22],  # soil
    [0.70, 0.66, 0.58],  # concrete
    [0.25, 0.22, 0.20],  # dark mulch
    [0.85, 0.83, 0.79],  # paper
    [0.55, 0.40, 0.28],  # wooden table
    [0.30, 0.38, 0.22],  # grass / foliage blur
]


def _background(rng, size):
    base = np.array(BACKGROUNDS[rng.integers(len(BACKGROUNDS))]) + rng.normal(0, 0.05, 3)
    tex = fractal_noise(rng, size)[..., None]
    return np.clip(base * (0.75 + 0.5 * tex), 0, 1)


def _paint_leaf(rng, img, size, shape, u, v, leaf, petiole, yellowing=0.0):
    green = np.array([rng.uniform(0.15, 0.32), rng.uniform(0.42, 0.66), rng.uniform(0.08, 0.25)])
    green = green * (1 - yellowing) + np.array([0.7, 0.68, 0.22]) * yellowing
    tex = fractal_noise(rng, size)[..., None]
    rgb = green * (0.8 + 0.4 * tex)
    if shape == "lobed":  # palmate venation: veins radiate from the base
        theta = np.arctan2(v, u)
        veins = np.exp(-((np.cos(theta * rng.choice([3, 5]) / 2) ** 2) * np.hypot(u, v) / 1.5))
    else:
        midrib = np.exp(-(v / 1.2) ** 2)
        side = (np.cos((u - np.abs(v) * 1.2) * rng.uniform(0.2, 0.35)) > 0.93) * np.exp(-np.abs(v) / 25)
        veins = np.clip(midrib + 0.6 * side, 0, 1)
    veins = veins[..., None]
    rgb = rgb * (1 - 0.35 * veins) + np.array([0.75, 0.85, 0.55]) * 0.35 * veins
    # Darken towards the edge for a little curvature / volume.
    img[leaf] = rgb[leaf]
    img[petiole] = green * 0.85
    return img


def generate_leaf(rng: np.random.Generator | None = None, size: int = IMG_SIZE,
                  disease: int | str | None = "random",
                  healthy_prob: float = 0.2, augment: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Generate one (image, mask) pair.

    disease: "random" (default), None for a healthy leaf, or a class index /
             class name from ``CLASSES`` to force a specific disease.

    Returns image float32 (size, size, 3) in [0, 1] and mask uint8 (size, size).
    """
    rng = rng or np.random.default_rng()
    if isinstance(disease, str) and disease != "random":
        disease = CLASSES.index(disease)

    img = _background(rng, size)
    label = np.zeros((size, size), np.uint8)

    # Occasionally a second, out-of-focus healthy leaf peeks in from behind.
    if augment and rng.random() < 0.25:
        shape = LEAF_SHAPES[rng.integers(len(LEAF_SHAPES))]
        a = rng.uniform(0, 2 * np.pi)
        offset = (0.55 * np.sin(a), 0.55 * np.cos(a))  # off to one side, partly out of frame
        back, bu, bv, bpet = _leaf_geometry(rng, size, shape, offset)
        img = _paint_leaf(rng, img, size, shape, bu, bv, back, bpet)
        img = _box_blur(img, 2)
        label[back | bpet] = LEAF

    shape = LEAF_SHAPES[rng.integers(len(LEAF_SHAPES))]
    leaf, u, v, petiole = _leaf_geometry(rng, size, shape)

    if disease == "random":
        diseases = [] if rng.random() < healthy_prob else list(
            rng.choice(DISEASES, size=1 + (rng.random() < 0.12), replace=False))
    elif disease is None:
        diseases = []
    else:
        diseases = [int(disease)]

    yellowing = rng.uniform(0, 0.25) if diseases else rng.uniform(0, 0.08)
    img = _paint_leaf(rng, img, size, shape, u, v, leaf, petiole, yellowing)
    label[leaf | petiole] = LEAF
    for d in diseases:
        DRAWERS[int(d)](rng, img, label, leaf, size)

    if augment:
        if rng.random() < 0.4:  # soft cast shadow from a half-plane
            a = rng.uniform(0, 2 * np.pi)
            yy, xx = np.mgrid[0:size, 0:size]
            s = (xx - size / 2) * np.cos(a) + (yy - size / 2) * np.sin(a) - rng.uniform(-30, 30)
            img = img * (1 - rng.uniform(0.25, 0.5) * (1 / (1 + np.exp(-s / 4))))[..., None]
        if rng.random() < 0.3:  # specular glare
            gy, gx = rng.uniform(0, size, 2)
            yy, xx = np.mgrid[0:size, 0:size]
            glare = np.exp(-((yy - gy) ** 2 + (xx - gx) ** 2) / (2 * rng.uniform(8, 20) ** 2))
            img = img + 0.25 * glare[..., None]
        if rng.random() < 0.3:
            img = _box_blur(img, 1)
        img = img * rng.uniform(0.75, 1.2)  # exposure
        img = img * (1 + rng.normal(0, 0.06, 3))  # white balance cast
        img = img + rng.normal(0, rng.uniform(0.01, 0.035), img.shape)  # sensor noise
    return np.clip(img, 0, 1).astype(np.float32), label


def generate_batch(n: int, rng: np.random.Generator | None = None, size: int = IMG_SIZE, **kw):
    """Return (images[N,3,H,W] float32, masks[N,H,W] int64) ready for PyTorch."""
    rng = rng or np.random.default_rng()
    imgs, masks = zip(*(generate_leaf(rng, size, **kw) for _ in range(n)))
    # Contiguous NCHW matters: non-contiguous input is ~3x slower on the MPS backend.
    return np.ascontiguousarray(np.stack(imgs).transpose(0, 3, 1, 2)), np.stack(masks).astype(np.int64)
