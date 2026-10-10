# Fine-tuning model:
# Wan2.2-TI2V-5B-Diffusers
# Wan (Wan Team et al., arXiv:2503.20314)
# Text-image-to-video diffusion model, fine-tuned with LoRA via ai-toolkit.
# Hugging Face: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageSequence

MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_DIR))

from prepare_data import prepare_dataset
from train import find_latest_lora, run_name, train_lora


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


def load_config():
    cfg = yaml.safe_load(open(Path(__file__).with_name("config.yaml")))
    return cfg


def _prompt_items(cfg):
    prompts = cfg.get("prompts")
    if not prompts:
        prompt_cfg = json.loads((MODEL_DIR / "prompts.json").read_text())
        prompts = prompt_cfg["prompts"]

    out = []
    for item in prompts:
        text = item.get("prompt") or item.get("text")
        out.append({
            "name": item["name"],
            "prompt": text,
        })
    return out


def _webp_to_mp4(webp_path, mp4_path, fps):
    from diffusers.utils import export_to_video

    img = Image.open(webp_path)
    frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(img)]
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(mp4_path), fps=fps)


def _format_path(template, **values):
    defaults = {
        "dataset": "default",
        "method": values.get("method", "my_method"),
        "track": "effectiveness",
        "model": "WAN2.2_5B",
    }
    defaults.update(values)
    return template.format(**defaults)


def _collect_toolkit_samples(cfg, repo_root: Path, method: str, samples_dir: Path):
    prompts = _prompt_items(cfg)
    work_dir = repo_root / _format_path(
        cfg["paths"]["work_dir"],
        dataset=cfg.get("dataset", "default"),
        method=method,
        track=cfg.get("track", "effectiveness"),
        model=cfg.get("eval_model", "WAN2.2_5B"),
    )
    out_dir = work_dir / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_files = sorted(samples_dir.glob("*.webp"), key=lambda p: p.stat().st_mtime)
    if not sample_files:
        raise FileNotFoundError(f"no ai-toolkit samples found under {samples_dir}")

    generated = []
    for idx, item in enumerate(prompts):
        src = sample_files[-len(prompts) + idx]
        mp4_path = out_dir / f"{item['name']}.mp4"
        _webp_to_mp4(src, mp4_path, cfg["inference"].get("fps", 24))
        print(f"[sample] {src.relative_to(repo_root)}")
        print(f"[video] {mp4_path.relative_to(repo_root)}")
        generated.append({
            "prompt": item["prompt"],
            "path": str(mp4_path.relative_to(repo_root)),
        })
    return generated


def _work_dir(cfg, repo_root: Path, method: str):
    return repo_root / _format_path(
        cfg["paths"]["work_dir"],
        dataset=cfg.get("dataset", "default"),
        method=method,
        track=cfg.get("track", "effectiveness"),
        model=cfg.get("eval_model", "WAN2.2_5B"),
    )


def _expected_generated(cfg, repo_root: Path, method: str):
    prompts = _prompt_items(cfg)
    out_dir = _work_dir(cfg, repo_root, method) / "generated"
    generated = []
    for item in prompts:
        mp4_path = out_dir / f"{item['name']}.mp4"
        if not mp4_path.exists():
            return None
        generated.append({
            "prompt": item["prompt"],
            "path": str(mp4_path.relative_to(repo_root)),
        })
    return generated


def _samples_dir(cfg, repo_root: Path, method: str):
    output_dir = _work_dir(cfg, repo_root, method) / cfg["paths"].get("output_dir", "output")
    return output_dir / run_name(cfg, method) / "samples"


def _find_lora_path(cfg, repo_root: Path, method: str):
    output_dir = _work_dir(cfg, repo_root, method) / cfg["paths"].get("output_dir", "output")
    try:
        return find_latest_lora(output_dir)
    except FileNotFoundError:
        return None


def _find_lora(cfg, repo_root: Path, method: str):
    path = _find_lora_path(cfg, repo_root, method)
    return str(path.relative_to(repo_root)) if path is not None else None


def run(eval_cfg, repo_root: Path):
    cfg = load_config()
    cfg["model_path"] = _model_path(repo_root)
    if eval_cfg.get("data_config"):
        cfg["data_config"] = eval_cfg["data_config"]
    if eval_cfg.get("seed") is not None:
        cfg["seed"] = eval_cfg["seed"]
    if eval_cfg.get("finetune"):
        ft_cfg = eval_cfg["finetune"]
        if ft_cfg.get("trigger_word"):
            cfg["trigger_word"] = ft_cfg["trigger_word"]
        if ft_cfg.get("caption"):
            cfg["caption"] = ft_cfg["caption"]
        if ft_cfg.get("negative_prompt"):
            cfg.setdefault("inference", {})["negative_prompt"] = ft_cfg["negative_prompt"]
        if ft_cfg.get("train"):
            cfg.setdefault("train", {}).update(ft_cfg["train"])
        if ft_cfg.get("prompts"):
            cfg["prompts"] = ft_cfg["prompts"]
    cfg["dataset"] = eval_cfg.get("dataset", "default")
    cfg["track"] = eval_cfg.get("work_track", eval_cfg.get("track", "effectiveness"))
    cfg["eval_model"] = eval_cfg.get("model", "WAN2.2_5B")
    method = eval_cfg.get("method", "my_method")
    work_method = eval_cfg.get("work_method", method)
    toolkit_root = repo_root / cfg["paths"].get("toolkit_root", "third_party/ai-toolkit")
    sys.path.insert(0, str(toolkit_root))

    existing_generated = _expected_generated(cfg, repo_root, work_method)
    if existing_generated is not None:
        return {
            "task": "finetune",
            "track": eval_cfg.get("track", "effectiveness"),
            "attack": eval_cfg.get("attack"),
            "model": eval_cfg.get("model", "WAN2.2_5B"),
            "method": method,
            "lora": _find_lora(cfg, repo_root, work_method),
            "generated": existing_generated,
        }

    prepared = prepare_dataset(
        cfg,
        repo_root,
        method,
        protected_root=eval_cfg.get("protected_root"),
        work_method=work_method,
        dataset=cfg["dataset"],
        track=cfg["track"],
        model=cfg["eval_model"],
    )

    samples_dir = _samples_dir(cfg, repo_root, work_method)
    if list(samples_dir.glob("*.webp")):
        generated = _collect_toolkit_samples(cfg, repo_root, work_method, samples_dir)
        return {
            "task": "finetune",
            "track": eval_cfg.get("track", "effectiveness"),
            "attack": eval_cfg.get("attack"),
            "model": eval_cfg.get("model", "WAN2.2_5B"),
            "method": method,
            "lora": _find_lora(cfg, repo_root, work_method),
            "generated": generated,
        }

    existing_lora = _find_lora_path(cfg, repo_root, work_method)
    if existing_lora is not None:
        print(f"[reuse] finetune LoRA checkpoint: {existing_lora.relative_to(repo_root)}")
        print("[reuse] launching ai-toolkit resume path for sampling; no extra FT steps run when checkpoint step >= train.steps")

    lora_path, samples_dir = train_lora(
        cfg,
        repo_root,
        work_method,
        prepared,
        skip_train=bool(eval_cfg.get("skip_train", False)),
    )
    generated = _collect_toolkit_samples(cfg, repo_root, work_method, samples_dir)

    return {
        "task": "finetune",
        "track": eval_cfg.get("track", "effectiveness"),
        "attack": eval_cfg.get("attack"),
        "model": eval_cfg.get("model", "WAN2.2_5B"),
        "method": method,
        "lora": str(lora_path.relative_to(repo_root)),
        "generated": generated,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="my_method")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parents[4]
    run({"method": args.method, "skip_train": args.skip_train}, repo_root)


if __name__ == "__main__":
    main()
