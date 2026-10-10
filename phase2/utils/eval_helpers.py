import gc
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from utils import io_video, metrics


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def load_model_runner(repo_root, task, model_name):
    path = Path(repo_root) / "eval" / task / "models" / model_name / "run.py"
    if not path.exists():
        raise FileNotFoundError(f"missing {task} model runner: {path}")
    spec = importlib.util.spec_from_file_location(f"phase2_{task}_{model_name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summary_is_complete(summary_path, repo_root):
    summary_path = Path(summary_path)
    if not summary_path.exists():
        return False
    summary = read_json(summary_path)
    for item in summary.get("generated", []):
        if not (Path(repo_root) / item["path"]).exists():
            return False
    return True


def skip_label(cfg):
    return cfg["track"] if not cfg.get("attack") else f"{cfg['track']}_{cfg['attack']}"


def runtime_summary(file_summary, cfg, task, summary_path, repo_root):
    return {
        **file_summary,
        "task": task,
        "track": cfg["track"],
        "attack": cfg.get("attack"),
        "model": cfg["model"],
        "method": cfg["method"],
        "summary_path": str(Path(summary_path).relative_to(repo_root)),
    }


def repo_path(repo_root, value, default=None):
    path = Path(value or default)
    return path if path.is_absolute() else Path(repo_root) / path


def repo_rel(repo_root, path):
    return str(Path(path).resolve().relative_to(Path(repo_root).resolve()))


def load_data_config(repo_root):
    data_config_path = repo_path(repo_root, "data_preparation/config.yaml")
    return data_config_path, load_yaml(data_config_path)


def data_root(repo_root, data_cfg, key, default):
    return repo_path(repo_root, data_cfg.get(key), default)


def clear_cuda_memory(label=None):
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"[cleanup] released CUDA cache after {label}")
    elif label:
        print(f"[cleanup] no initialized CUDA context after {label}")


def track_label(track, attack=None):
    return f"{track}_{attack}" if attack else track


def clip_paths(repo_root, data_cfg, method, clip, protected_root_override=None):
    preprocessed_root = data_root(repo_root, data_cfg, "output_dir", "data/preprocessed")
    protected_root = Path(protected_root_override) if protected_root_override else (
        data_root(repo_root, data_cfg, "protected_root", "data/protected") / method / "submission"
    )
    if not protected_root.is_absolute():
        protected_root = Path(repo_root) / protected_root
    ident, vid = clip["id"], clip["vid"]
    original = preprocessed_root / ident / f"{vid}.mp4"
    protected = protected_root / ident / f"{vid}.mp4"
    return original, protected


def attack_avg3(video):
    v = video.astype(np.int16)
    out = video.copy()
    out[1:-1] = ((v[:-2] + v[1:-1] + v[2:]) // 3).astype(np.uint8)
    out[0] = ((v[0] + v[1] + v[2]) // 3).astype(np.uint8)
    out[-1] = ((v[-3] + v[-2] + v[-1]) // 3).astype(np.uint8)
    return out


ATTACKS = {
    "avg3": attack_avg3,
}


def ensure_attack_dataset(repo_root, data_cfg, method, attack):
    if attack not in ATTACKS:
        sys.exit(f"unknown attack: {attack}")

    protected_root = data_root(repo_root, data_cfg, "protected_root", "data/protected")
    attacked_root = data_root(repo_root, data_cfg, "attacked_root", "data/attacked")
    src_root = protected_root / method / "submission"
    out_root = attacked_root / method / attack / "submission"

    for ident in data_cfg["identities"]:
        ident_id = ident["id"]
        for clip in ident["clips"]:
            src = src_root / ident_id / f"{clip['name']}.mp4"
            dst = out_root / ident_id / f"{clip['name']}.mp4"
            if dst.exists():
                continue
            if not src.exists():
                sys.exit(f"missing {src}; run protect/run.py --method {method} first")
            video = io_video.load_mp4(src)
            attacked = ATTACKS[attack](video)
            io_video.save_mp4(attacked, dst, lossless=True)
            print(f"[attack] {attack}: {dst.relative_to(repo_root)}")
    return out_root


def compute_invisibility(repo_root, data_cfg, method, budget):
    max_eps = 0
    psnr_values = []
    for ident in data_cfg["identities"]:
        ident_id = ident["id"]
        for clip in ident["clips"]:
            clip_ref = {"id": ident_id, "vid": clip["name"]}
            original_path, protected_path = clip_paths(repo_root, data_cfg, method, clip_ref)
            if not original_path.exists():
                sys.exit(f"missing {original_path}; run data_preparation/preprocess.py first")
            if not protected_path.exists():
                sys.exit(f"missing {protected_path}; run protect/run.py --method {method} first")
            orig = io_video.load_mp4(original_path)
            prot = io_video.load_mp4(protected_path)
            eps = int(np.abs(orig.astype(np.int16) - prot.astype(np.int16)).max())
            psnr = metrics.psnr(orig, prot)
            max_eps = max(max_eps, eps)
            psnr_values.append(psnr)
            del orig, prot
            torch.cuda.empty_cache()
            gc.collect()
    return {
        "eps": max_eps,
        "psnr": float(np.mean(psnr_values)) if psnr_values else None,
        "budget_linf": int(budget),
        "within_budget": bool(max_eps <= int(budget)),
    }


def face_reference_embedding(repo_root, cfg, data_cfg, method):
    ref = cfg["i2v"]["generation"]["reference_image"]
    original_path, _ = clip_paths(repo_root, data_cfg, method, ref)
    if not original_path.exists():
        sys.exit(f"missing face similarity reference video: {original_path}")
    video = io_video.load_mp4(original_path)
    emb, _ = metrics.face_embedding(video[int(ref.get("frame", 0))])
    del video
    return emb


def score_generation_summary(repo_root, summary, ref_emb):
    scored = []
    for item in summary.get("generated", []):
        video = io_video.load_mp4(Path(repo_root) / item["path"])
        scores = {
            "face_detection": metrics.face_detection(video),
            "face_similarity": metrics.face_similarity(video, ref_emb) if ref_emb is not None else None,
            "image_quality": metrics.brisque(video),
        }
        if item.get("supports_prompt", True):
            scores["prompt_following"] = metrics.prompt_following(video, item["prompt"])
        else:
            scores["prompt_following"] = None
        scored.append(scores)
        del video
        torch.cuda.empty_cache()
        gc.collect()
    if len(scored) == 1:
        return scored[0]
    return {Path(item["path"]).stem: scores for item, scores in zip(summary.get("generated", []), scored)}


def put_track_result(results, summary, scored):
    task = summary["task"]
    track = summary["track"]
    model = summary["model"]
    attack = summary.get("attack")
    results.setdefault(task, {})
    if track == "robustness":
        results[task].setdefault(track, {}).setdefault(attack, {})[model] = scored
    else:
        results[task].setdefault(track, {})[model] = scored


def format_score(value, digits=3):
    if value is None:
        return "None"
    return f"{value:.{digits}f}"


def format_metric_scores(scored):
    if not isinstance(scored, dict) or any(isinstance(v, dict) for v in scored.values()):
        return json.dumps(scored, default=str)
    return (
        f"face_detection={format_score(scored.get('face_detection'))}  "
        f"face_similarity={format_score(scored.get('face_similarity'))}  "
        f"prompt_following={format_score(scored.get('prompt_following'))}  "
        f"image_quality={format_score(scored.get('image_quality'), 1)}"
    )


def print_metric_section(summary, scored):
    label = track_label(summary["track"], summary.get("attack"))
    print(f"\n== {summary['task']}/{label} ==")
    prefix = f"[{summary['model']}] "
    if summary["track"] == "robustness" and summary.get("attack"):
        prefix = f"[{summary['model']}/{summary['attack']}] "
    print(f"  {prefix}{format_metric_scores(scored)}")
