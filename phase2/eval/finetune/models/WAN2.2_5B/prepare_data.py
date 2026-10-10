from pathlib import Path

import yaml


def _link_or_copy(src, dst):
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def _repo_path(repo_root: Path, value):
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _format_path(template, **values):
    defaults = {
        "dataset": "default",
        "method": values.get("work_method") or values.get("method", "my_method"),
        "track": "effectiveness",
        "model": "WAN2.2_5B",
    }
    defaults.update(values)
    return template.format(**defaults)


def prepare_dataset(cfg, repo_root: Path, method: str, protected_root=None, work_method=None, dataset="default", track="effectiveness", model="WAN2.2_5B"):
    model_dir = Path(__file__).resolve().parent
    work_key = work_method or method
    work_dir = repo_root / _format_path(
        cfg["paths"]["work_dir"],
        dataset=dataset,
        method=method,
        work_method=work_key,
        track=track,
        model=model,
    )
    dataset_dir = work_dir / cfg["paths"].get("dataset_dir", "dataset")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(dataset_dir.glob("*.mp4")) + list(dataset_dir.glob("*.txt")):
        stale.unlink()

    data_cfg_path = repo_root / cfg.get("data_config", "data_preparation/config.yaml")
    if not data_cfg_path.exists():
        data_cfg_path = repo_root / "data_preparation" / "config.yaml"
    data_cfg = yaml.safe_load(open(data_cfg_path))
    if len(data_cfg["identities"]) != 1:
        raise ValueError("finetune expects exactly one identity in the data config")
    ident_cfg = data_cfg["identities"][0]
    identity = ident_cfg["id"]
    if protected_root:
        protected_dir = _repo_path(repo_root, protected_root) / identity
    else:
        protected_dir = repo_root / "data" / "protected" / method / "submission" / identity
    if not protected_dir.exists():
        raise FileNotFoundError(f"missing {protected_dir}; run protect/run.py --method {method} first")

    expected = [protected_dir / f"{clip['name']}.mp4" for clip in ident_cfg["clips"]]
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing {missing[0]}; run protect/run.py --method {method} first")

    caption = cfg.get("caption") or f"{cfg.get('trigger_word', 'p3r5on')} is talking to the camera"
    videos = expected

    for src in videos:
        dst = dataset_dir / src.name
        _link_or_copy(src, dst)
        (dataset_dir / f"{src.stem}.txt").write_text(caption + "\n")
        print(f"[dataset] {dst.relative_to(repo_root)}")

    return {
        "model_dir": model_dir,
        "work_dir": work_dir,
        "dataset_dir": dataset_dir,
        "num_videos": len(videos),
    }
