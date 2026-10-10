import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils import eval_helpers as eh


def _run_i2v_tracks(cfg, data_cfg, method, data_config_path):
    i2v_cfg = cfg.get("i2v", {})
    generation = i2v_cfg["generation"]
    clip = generation["reference_image"]
    prompt = generation["prompt"]
    seed = cfg.get("seed", 42)
    generated_root = eh.data_root(REPO_ROOT, data_cfg, "generated_root", "data/generated")
    clean_original, clean_protected = eh.clip_paths(REPO_ROOT, data_cfg, method, clip)
    task_mod = importlib.import_module("eval.i2v.run")
    summaries = []

    for track, track_cfg in i2v_cfg.get("tracks", {}).items():
        if not (track_cfg and track_cfg.get("enabled", False)):
            continue
        if track == "robustness":
            for attack in track_cfg.get("attacks", []):
                label = eh.track_label(track, attack)
                attacked_root = eh.ensure_attack_dataset(REPO_ROOT, data_cfg, method, attack)
                _, input_video = eh.clip_paths(REPO_ROOT, data_cfg, method, clip, protected_root_override=attacked_root)
                for model in track_cfg.get("models", []):
                    print(f"\n== generation i2v/{label} ==")
                    out_dir = generated_root / method / "i2v" / label / model / clip["id"] / clip["vid"]
                    summaries.append(task_mod.run_track({
                        "model": model,
                        "method": method,
                        "track": track,
                        "attack": attack,
                        "clip": clip,
                        "prompt": prompt,
                        "generation": generation,
                        "seed": seed,
                        "data_config": eh.repo_rel(REPO_ROOT, data_config_path),
                        "original": clean_original,
                        "protected": clean_protected,
                        "input_video": input_video,
                        "output_dir": out_dir,
                    }, REPO_ROOT))
        else:
            for model in track_cfg.get("models", []):
                print(f"\n== generation i2v/{track} ==")
                out_dir = generated_root / method / "i2v" / track / model / clip["id"] / clip["vid"]
                summaries.append(task_mod.run_track({
                    "model": model,
                    "method": method,
                    "track": track,
                    "clip": clip,
                    "prompt": prompt,
                    "generation": generation,
                    "seed": seed,
                    "data_config": eh.repo_rel(REPO_ROOT, data_config_path),
                    "original": clean_original,
                    "protected": clean_protected,
                    "input_video": clean_protected,
                    "output_dir": out_dir,
                }, REPO_ROOT))
    return summaries


def _run_finetune_tracks(cfg, data_cfg, method, data_config_path):
    finetune_cfg = cfg.get("finetune", {})
    i2v_clip = cfg.get("i2v", {}).get("generation", {}).get("reference_image", {})
    generated_root = eh.data_root(REPO_ROOT, data_cfg, "generated_root", "data/generated")
    clean_root = eh.data_root(REPO_ROOT, data_cfg, "protected_root", "data/protected") / method / "submission"
    clean_original, clean_protected = eh.clip_paths(REPO_ROOT, data_cfg, method, i2v_clip)
    task_mod = importlib.import_module("eval.finetune.run")
    summaries = []

    for track, track_cfg in finetune_cfg.get("tracks", {}).items():
        if not (track_cfg and track_cfg.get("enabled", False)):
            continue
        if track == "robustness":
            for attack in track_cfg.get("attacks", []):
                protected_root = eh.ensure_attack_dataset(REPO_ROOT, data_cfg, method, attack)
                _, protected_clip = eh.clip_paths(REPO_ROOT, data_cfg, method, i2v_clip, protected_root_override=protected_root)
                for model in track_cfg.get("models", []):
                    label = eh.track_label(track, attack)
                    print(f"\n== generation finetune/{label} ==")
                    out_dir = generated_root / method / "finetune" / label / model
                    summaries.append(task_mod.run_track({
                        "model": model,
                        "method": method,
                        "dataset": data_cfg.get("dataset_name", "default"),
                        "work_method": method,
                        "track": track,
                        "work_track": label,
                        "attack": attack,
                        "seed": cfg.get("seed", 42),
                        "data_config": eh.repo_rel(REPO_ROOT, data_config_path),
                        "finetune": {
                            "trigger_word": finetune_cfg.get("trigger_word"),
                            "caption": finetune_cfg.get("caption"),
                            "negative_prompt": finetune_cfg.get("negative_prompt"),
                            "train": finetune_cfg.get("train", {}),
                            "prompts": finetune_cfg.get("prompts", []),
                        },
                        "protected_root": eh.repo_rel(REPO_ROOT, protected_root),
                        "original": clean_original,
                        "protected": protected_clip,
                        "clip": i2v_clip,
                        "summary_path": out_dir / "generation_summary.json",
                    }, REPO_ROOT))
        else:
            for model in track_cfg.get("models", []):
                print(f"\n== generation finetune/{track} ==")
                out_dir = generated_root / method / "finetune" / track / model
                summaries.append(task_mod.run_track({
                    "model": model,
                    "method": method,
                    "dataset": data_cfg.get("dataset_name", "default"),
                    "work_method": method,
                    "track": track,
                    "seed": cfg.get("seed", 42),
                    "data_config": eh.repo_rel(REPO_ROOT, data_config_path),
                    "finetune": {
                        "trigger_word": finetune_cfg.get("trigger_word"),
                        "caption": finetune_cfg.get("caption"),
                        "negative_prompt": finetune_cfg.get("negative_prompt"),
                        "train": finetune_cfg.get("train", {}),
                        "prompts": finetune_cfg.get("prompts", []),
                    },
                    "protected_root": eh.repo_rel(REPO_ROOT, clean_root),
                    "original": clean_original,
                    "protected": clean_protected,
                    "clip": i2v_clip,
                    "summary_path": out_dir / "generation_summary.json",
                }, REPO_ROOT))
    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="my_method")
    ap.add_argument("--config", default=str(REPO_ROOT / "eval" / "config.yaml"))
    args = ap.parse_args()

    cfg = eh.load_yaml(args.config)
    data_config_path, data_cfg = eh.load_data_config(REPO_ROOT)
    dataset = data_cfg.get("dataset_name", "default")
    i2v_clip = cfg["i2v"]["generation"]["reference_image"]
    original, protected = eh.clip_paths(REPO_ROOT, data_cfg, args.method, i2v_clip)
    if not original.exists():
        sys.exit(f"missing {original}; run data_preparation/preprocess.py first")
    if not protected.exists():
        sys.exit(f"missing {protected}; run protect/run.py --method {args.method} first")

    print(f"[dataset] {dataset}")
    print(f"[method]  {args.method}")
    print(f"[config]  {Path(args.config).resolve().relative_to(REPO_ROOT)}")

    summaries = []
    i2v_summaries = _run_i2v_tracks(cfg, data_cfg, args.method, data_config_path)
    summaries.extend(i2v_summaries)
    if i2v_summaries:
        eh.clear_cuda_memory("i2v")
    summaries.extend(_run_finetune_tracks(cfg, data_cfg, args.method, data_config_path))

    print("\n== metrics ==")
    invisibility = eh.compute_invisibility(REPO_ROOT, data_cfg, args.method, cfg.get("budget", {}).get("linf", 16))
    print("\n== invisibility ==")
    print(
        "  "
        f"eps={invisibility['eps']}  "
        f"psnr={eh.format_score(invisibility.get('psnr'), 2)}  "
        f"within_budget={invisibility['within_budget']}"
    )

    results = {
        "method": args.method,
        "dataset": dataset,
        "config": {
            "data_config": eh.repo_rel(REPO_ROOT, data_config_path),
            "seed": cfg.get("seed", 42),
            "i2v_reference_image": i2v_clip,
            "i2v_prompt": cfg["i2v"]["generation"]["prompt"],
            "i2v_generation": cfg["i2v"]["generation"],
            "budget_linf": cfg.get("budget", {}).get("linf", 16),
            "face_similarity_reference_image": i2v_clip,
        },
        "invisibility": invisibility,
    }

    ref_emb = eh.face_reference_embedding(REPO_ROOT, cfg, data_cfg, args.method)
    for summary in summaries:
        scored = eh.score_generation_summary(REPO_ROOT, summary, ref_emb)
        eh.put_track_result(results, summary, scored)
        eh.print_metric_section(summary, scored)

    results_root = eh.data_root(REPO_ROOT, data_cfg, "results_root", "data/results")
    out_path = results_root / args.method / "results.json"
    eh.write_json(results, out_path)
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
