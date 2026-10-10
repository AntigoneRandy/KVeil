import numpy as np

EPS = 16
SEED = 42


def protect(videos: dict) -> dict:
    """Add uniform integer noise in [-EPS, +EPS] to every frame of every clip."""
    rng = np.random.default_rng(SEED)
    out = {}
    for n, v in videos.items():
        noise = rng.integers(-EPS, EPS + 1, size=v.shape, dtype=np.int16)
        out[n] = np.clip(v.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out
