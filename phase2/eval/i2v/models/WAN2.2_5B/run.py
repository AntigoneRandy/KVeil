# I2V model:
# Wan2.2-TI2V-5B-Diffusers
# Wan (Wan Team et al., arXiv:2503.20314)
# Text-image-to-video diffusion model.
# Hugging Face: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers

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


def _patch_scheduler_device(scheduler):
    original_step = scheduler.step

    def step(model_output, timestep, sample, *args, **kwargs):
        device = sample.device
        for attr in ("sigmas", "timesteps"):
            value = getattr(scheduler, attr, None)
            if torch.is_tensor(value) and value.device != device:
                setattr(scheduler, attr, value.to(device))
        if torch.is_tensor(timestep) and timestep.device != device:
            timestep = timestep.to(device)
        return original_step(model_output, timestep, sample, *args, **kwargs)

    scheduler.step = step


def _model_path(repo_root):
    default = repo_root / "models" / "Wan2.2-TI2V-5B-Diffusers"
    if default.exists():
        return str(default)
    path = os.environ.get("WAN_MODEL_PATH")
    if path:
        return path
    raise RuntimeError(
        f"WAN model not found at {default}. "
        "Download it there or set WAN_MODEL_PATH."
    )


def generate(protected_video, output_dir, prompt, frame_idx, seed, generation, repo_root):
    from diffusers import WanImageToVideoPipeline
    from utils import io_video

    cfg = generation
    model_path = _model_path(repo_root)
    frame = _load_first_frame(protected_video, frame_idx)
    image = Image.fromarray(frame).resize((cfg["width"], cfg["height"]), Image.LANCZOS)

    print(f"[load] {model_path}")
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload(device="cuda")
    _patch_scheduler_device(pipe.scheduler)
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
    io_video.save_generated_mp4(frames, out_path, fps=cfg.get("fps", 24))
    print(f"[video] {out_path.relative_to(repo_root)}")
    return [{"prompt": prompt, "path": str(out_path.relative_to(repo_root))}]
