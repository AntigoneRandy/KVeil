from pathlib import Path

from utils import eval_helpers as eh


def run_track(cfg, repo_root: Path):
    summary_path = Path(cfg["summary_path"])
    if not summary_path.is_absolute():
        summary_path = repo_root / summary_path
    if eh.summary_is_complete(summary_path, repo_root):
        print(f"[skip] finetune {eh.skip_label(cfg)}")
        summary = eh.read_json(summary_path)
        file_summary = {k: v for k, v in summary.items() if k in {"lora", "generated"}}
        return eh.runtime_summary(file_summary, cfg, "finetune", summary_path, repo_root)

    runner = eh.load_model_runner(repo_root, "finetune", cfg["model"])
    summary = runner.run(cfg, repo_root)
    file_summary = {k: v for k, v in summary.items() if k in {"lora", "generated"}}
    eh.write_json(file_summary, summary_path)
    print(f"[summary] {summary_path.relative_to(repo_root)}")
    return eh.runtime_summary(file_summary, cfg, "finetune", summary_path, repo_root)


def run(cfg, repo_root: Path):
    model_name = cfg.get("model", "WAN2.2_5B")
    runner = eh.load_model_runner(repo_root, "finetune", model_name)
    return runner.run(cfg, repo_root)
