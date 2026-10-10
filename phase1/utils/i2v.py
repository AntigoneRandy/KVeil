import os

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL_PATH = os.environ.get(
    "WAN_MODEL_PATH", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
)
TARGET_W, TARGET_H = 640, 480
DEFAULT_NEGATIVE = "low quality, blurry, distorted, deformed face"


def load_wan_i2v(model_path=None, device="cuda"):
    """Load Wan2.2-TI2V-5B in bf16 with CPU offload."""
    from diffusers import WanImageToVideoPipeline

    pipe = WanImageToVideoPipeline.from_pretrained(
        model_path or DEFAULT_MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload(device=device)
    return pipe


def run_i2v(
    pipe,
    first_frame,
    prompt,
    *,
    seed=42,
    num_frames=49,
    num_inference_steps=25,
    guidance_scale=5.0,
    negative_prompt=DEFAULT_NEGATIVE,
):
    """Generate a video from one conditioning frame and a text prompt."""
    if first_frame.shape[:2] != (TARGET_H, TARGET_W):
        first_frame = resize_crop(first_frame, TARGET_W, TARGET_H)
    img = Image.fromarray(first_frame)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    out = pipe(
        image=img,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=TARGET_H,
        width=TARGET_W,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=gen,
    ).frames[0]
    return _to_uint8_array(out)


def resize_crop(frame_uint8, target_w, target_h):
    """Proportional resize to cover (target_w, target_h), then center-crop."""
    img = Image.fromarray(frame_uint8)
    w, h = img.size
    scale = max(target_w / w, target_h / h)
    new_w, new_h = round(w * scale), round(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return np.array(img.crop((left, top, left + target_w, top + target_h)))


def _to_uint8_array(frames):
    """Normalize pipeline output to uint8 numpy (T, H, W, 3)."""
    if hasattr(frames[0], "mode"):
        return np.stack([np.array(f) for f in frames])
    arr = np.stack(frames) if isinstance(frames, list) else np.asarray(frames)
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr
