from pathlib import Path

import numpy as np
import torch

SCHEMA = {
    "shape": (113, 480, 640, 3),
    "dtype": torch.uint8,
    "value_range": (0, 255),
    "fps": 30,
}

LOSSLESS_PROBE = {
    "codec_name": "h264",
    "pix_fmt": "gbrp",
}


def load_mp4(path):
    """Read an mp4 into uint8 (T, H, W, 3)."""
    from decord import VideoReader, cpu

    vr = VideoReader(str(path), ctx=cpu(0))
    return vr.get_batch(list(range(len(vr)))).asnumpy()


def _validate(frames, path):
    if frames.dtype != SCHEMA["dtype"]:
        raise ValueError(
            f"{path}: dtype must be {SCHEMA['dtype']}, got {frames.dtype}"
        )
    if tuple(frames.shape) != SCHEMA["shape"]:
        raise ValueError(
            f"{path}: shape must be {SCHEMA['shape']}, got {tuple(frames.shape)}"
        )


def save_mp4(frames, path, lossless=True, fps=None):
    import subprocess

    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    _validate(torch.from_numpy(frames), path)
    fps = fps if fps is not None else SCHEMA["fps"]
    h, w = frames.shape[1:3]
    if lossless:
        enc = ["-c:v", "libx264rgb", "-crf", "0", "-preset", "veryslow",
               "-color_primaries", "bt709", "-color_trc", "iec61966-2-1",
               "-colorspace", "rgb", "-color_range", "pc"]
    else:
        enc = ["-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-preset", "medium",
               "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        *enc, str(path),
    ]
    p = subprocess.run(cmd, input=np.ascontiguousarray(frames, dtype=np.uint8).tobytes())
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed encoding {path}")
