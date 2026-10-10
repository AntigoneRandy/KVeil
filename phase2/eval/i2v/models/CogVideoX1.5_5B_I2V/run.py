# I2V model:
# CogVideoX1.5-5B-I2V
# CogVideoX (Yang et al., arXiv:2408.06072)
# Text-image-to-video diffusion model.
# Hugging Face: https://huggingface.co/zai-org/CogVideoX1.5-5B-I2V

import os

import numpy as np
import torch
from PIL import Image


def _to_uint8_array(frames):
    if hasattr(frames[0], "mode"):
        return np.stack([np.array(f) for f in frames])
    arr = np.stack(frames) if isinstance(frames, list) else np.asarray(frames)
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr


def _load_first_frame(video_path, frame_idx):
    from decord import VideoReader, cpu

    vr = VideoReader(str(video_path), ctx=cpu(0))
    return vr[frame_idx].asnumpy()


def _resize_crop(frame_uint8, target_w, target_h):
    image = Image.fromarray(frame_uint8)
    w, h = image.size
    scale = max(target_w / w, target_h / h)
    new_w, new_h = round(w * scale), round(h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return image.crop((left, top, left + target_w, top + target_h))


def _model_path(repo_root):
    default = repo_root / "models" / "CogVideoX1.5-5B-I2V"
    if default.exists():
        return str(default)
    path = os.environ.get("COGVIDEOX_I2V_MODEL_PATH")
    if path:
        return path
    raise RuntimeError(
        f"CogVideoX I2V model not found at {default}. "
        "Download it there or set COGVIDEOX_I2V_MODEL_PATH."
    )


def generate(protected_video, output_dir, prompt, frame_idx, seed, generation, repo_root):
    from diffusers import CogVideoXImageToVideoPipeline
    from utils import io_video

    cfg = generation
    model_path = _model_path(repo_root)

    frame = _load_first_frame(protected_video, frame_idx)
    image = _resize_crop(frame, cfg["width"], cfg["height"])

    print(f"[load] {model_path}")
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_sequential_cpu_offload()
    if hasattr(pipe, "vae"):
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

    gen = torch.Generator(device="cuda").manual_seed(seed)
    out = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=cfg.get("negative_prompt", ""),
        height=cfg["height"],
        width=cfg["width"],
        num_frames=cfg["num_frames"],
        num_inference_steps=cfg["num_inference_steps"],
        guidance_scale=cfg["guidance_scale"],
        generator=gen,
    ).frames[0]
    frames = _to_uint8_array(out)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "talking.mp4"
    io_video.save_generated_mp4(frames, out_path, fps=cfg.get("fps", 8))
    print(f"[video] {out_path.relative_to(repo_root)}")
    return [{"prompt": prompt, "path": str(out_path.relative_to(repo_root))}]
