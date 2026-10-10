import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils import eval_helpers as eh


def _generated_item(output_dir, prompt, repo_root):
    out_path = output_dir / "talking.mp4"
    item = {
        "prompt": prompt,
        "path": str(out_path.relative_to(repo_root)),
    }
    return item if out_path.exists() else None


def _generate_and_write(cfg, repo_root: Path):
    model_name = cfg["model"]
    clip = cfg["clip"]
    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    summary_path = output_dir / "generation_summary.json"
    input_video = Path(cfg["input_video"])
    if not input_video.is_absolute():
        input_video = repo_root / input_video
    if not input_video.exists():
        raise FileNotFoundError(f"missing input video: {input_video}")

    runner = eh.load_model_runner(repo_root, "i2v", model_name)
    generated = runner.generate(
        protected_video=input_video,
        output_dir=output_dir,
        prompt=cfg["prompt"],
        frame_idx=int(clip.get("frame", 0)),
        seed=int(cfg.get("seed", 42)),
        generation=cfg["generation"],
        repo_root=repo_root,
    )
    file_summary = {
        "input_video": str(input_video.relative_to(repo_root)),
        "generated": generated,
    }
    eh.write_json(file_summary, summary_path)
    print(f"[summary] {summary_path.relative_to(repo_root)}")
    return file_summary


def _generate_in_subprocess(cfg, repo_root: Path, output_dir: Path, summary_path: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "_generation_request.json"
    eh.write_json({
        "repo_root": str(repo_root),
        "cfg": cfg,
    }, request_path)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--generate-config",
        str(request_path),
    ]
    print("[i2v-subprocess] " + " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=str(repo_root), check=True)
    finally:
        request_path.unlink(missing_ok=True)
    return eh.read_json(summary_path)


def run_track(cfg, repo_root: Path):
    model_name = cfg["model"]
    clip = cfg["clip"]
    generation = cfg["generation"]
    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    summary_path = output_dir / "generation_summary.json"
    existing_item = _generated_item(output_dir, cfg["prompt"], repo_root)

    if eh.summary_is_complete(summary_path, repo_root):
        print(f"[skip] i2v {eh.skip_label(cfg)}")
        summary = eh.read_json(summary_path)
        file_summary = {
            "input_video": summary.get("input_video"),
            "generated": summary.get("generated", []),
        }
        return eh.runtime_summary(file_summary, cfg, "i2v", summary_path, repo_root)

    input_video = Path(cfg["input_video"])
    if not input_video.is_absolute():
        input_video = repo_root / input_video

    if existing_item is not None:
        print(f"[skip] i2v {eh.skip_label(cfg)}")
        file_summary = {
            "input_video": str(input_video.relative_to(repo_root)),
            "generated": [existing_item],
        }
        eh.write_json(file_summary, summary_path)
        print(f"[summary] {summary_path.relative_to(repo_root)}")
        return eh.runtime_summary(file_summary, cfg, "i2v", summary_path, repo_root)

    file_summary = _generate_in_subprocess(cfg, repo_root, output_dir, summary_path)
    return eh.runtime_summary(file_summary, cfg, "i2v", summary_path, repo_root)


def run(cfg, repo_root: Path):
    model_name = cfg.get("model", "WAN2.2_5B")
    method = cfg.get("method", "my_method")
    generation = cfg["generation"]
    clip = generation["reference_image"]
    ident, vid = clip["id"], clip["vid"]
    original = repo_root / "data" / "preprocessed" / ident / f"{vid}.mp4"
    protected = repo_root / "data" / "protected" / method / "submission" / ident / f"{vid}.mp4"
    out_dir = repo_root / "data" / "generated" / "i2v" / method / model_name / ident / vid
    return run_track({
        "model": model_name,
        "method": method,
        "track": "effectiveness",
        "clip": clip,
        "prompt": generation["prompt"],
        "generation": generation,
        "seed": cfg.get("seed", 42),
        "original": original,
        "protected": protected,
        "input_video": protected,
        "output_dir": out_dir,
    }, repo_root)


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate-config", required=True)
    args = ap.parse_args()
    request = eh.read_json(args.generate_config)
    _generate_and_write(request["cfg"], Path(request["repo_root"]))


if __name__ == "__main__":
    _main()
