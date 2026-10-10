import subprocess
from pathlib import Path
from string import Template

import yaml


def _format_path(template, **values):
    defaults = {
        "dataset": "default",
        "method": values.get("method", "my_method"),
        "track": "effectiveness",
        "model": "WAN2.2_5B",
    }
    defaults.update(values)
    return template.format(**defaults)


def _yaml_value(value):
    return yaml.safe_dump(value, default_flow_style=True).strip()


def _sample_prompts(cfg):
    prompts = cfg.get("prompts") or []
    if not prompts:
        return [cfg.get("caption", "the p3r5on is talking to the camera")]
    return [item.get("prompt") or item.get("text") for item in prompts]


def run_name(cfg, method):
    return "_".join([
        cfg.get("dataset", "default"),
        cfg.get("eval_model", "WAN2.2_5B"),
        method,
        cfg.get("track", "effectiveness"),
    ])


def render_train_config(cfg, repo_root: Path, method: str, prepared):
    model_dir = Path(__file__).resolve().parent
    work_dir = prepared["work_dir"]
    output_dir = work_dir / cfg["paths"].get("output_dir", "output")
    rendered = work_dir / cfg["paths"].get("rendered_config", "train_config.yaml")
    output_dir.mkdir(parents=True, exist_ok=True)

    train = cfg["train"]
    inference = cfg["inference"]
    steps = train["steps"]
    sample_every = train.get("sample_every") or steps
    save_every = train.get("save_every") or steps
    name = run_name(cfg, method)
    mapping = {
        "run_name": name,
        "output_dir": str(output_dir),
        "dataset_dir": str(prepared["dataset_dir"]),
        "device": cfg.get("device", "cuda:0"),
        "dtype": cfg.get("dtype", "bf16"),
        "trigger_word": cfg.get("trigger_word", "p3r5on"),
        "caption": cfg.get("caption", "the p3r5on is talking to the camera"),
        "model_path": cfg["model_path"],
        "arch": cfg.get("arch", "wan22_5b"),
        "rank": train.get("rank", 32),
        "alpha": train.get("alpha", 32),
        "save_every": save_every,
        "num_frames": train.get("num_frames", 49),
        "resolution": _yaml_value(train.get("resolution", [640])),
        "batch_size": train.get("batch_size", 1),
        "steps": steps,
        "gradient_accumulation": train.get("gradient_accumulation", 1),
        "lr": train.get("lr", "1e-4"),
        "skip_first_sample": str(train.get("skip_first_sample", True)).lower(),
        "disable_sampling": str(train.get("disable_sampling", True)).lower(),
        "cache_text_embeddings": str(train.get("cache_text_embeddings", True)).lower(),
        "cache_latents_to_disk": str(train.get("cache_latents_to_disk", True)).lower(),
        "sample_every": sample_every,
        "sample_width": inference.get("width", 640),
        "sample_height": inference.get("height", 480),
        "sample_num_frames": inference.get("num_frames", 113),
        "sample_fps": inference.get("fps", 24),
        "sample_prompts": _yaml_value(_sample_prompts(cfg)),
        "sample_seed": cfg.get("seed", 42),
        "guidance_scale": inference.get("guidance_scale", 4.0),
        "sample_steps": inference.get("num_inference_steps", 30),
        "negative_prompt": inference.get("negative_prompt", ""),
        "network_multiplier": inference.get("network_multiplier", 1.0),
    }

    text = Template((model_dir / "train_config.template.yaml").read_text()).substitute(mapping)
    rendered.parent.mkdir(parents=True, exist_ok=True)
    rendered.write_text(text)
    print(f"[train-config] {rendered.relative_to(repo_root)}")
    return rendered


def find_latest_lora(output_dir: Path):
    candidates = sorted(output_dir.rglob("*.safetensors"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no .safetensors LoRA found under {output_dir}")
    return candidates[-1]


def train_lora(cfg, repo_root: Path, method: str, prepared, skip_train=False):
    rendered = render_train_config(cfg, repo_root, method, prepared)
    output_dir = prepared["work_dir"] / cfg["paths"].get("output_dir", "output")
    existing_lora = None
    try:
        existing_lora = find_latest_lora(output_dir)
    except FileNotFoundError:
        pass

    if not skip_train:
        toolkit_root = repo_root / cfg["paths"].get("toolkit_root", "third_party/ai-toolkit")
        cmd = ["python", str(toolkit_root / "run.py"), str(rendered)]
        if existing_lora is None:
            print("[train] " + " ".join(cmd))
        else:
            print("[infer] " + " ".join(cmd))
        subprocess.run(cmd, cwd=str(toolkit_root), check=True)
    lora_path = find_latest_lora(output_dir)
    print(f"[lora] {lora_path.relative_to(repo_root)}")
    return lora_path, output_dir / run_name(cfg, method) / "samples"
