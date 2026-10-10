import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import eval_helpers as eh, io_video
import scorer as SC


def compute_invisibility(data_cfg, method):
    per_clip = {}
    for ident in data_cfg["identities"]:
        for clip in ident["clips"]:
            ref = {"id": ident["id"], "vid": clip["name"]}
            original, protected = eh.clip_paths(REPO_ROOT, data_cfg, method, ref)
            for p in (original, protected):
                if not p.exists():
                    sys.exit(f"missing {p}")
            fps = data_cfg.get("target", {}).get("fps", 24)
            per_clip[clip["name"]] = SC.invisibility(
                io_video.load_mp4(original), io_video.load_mp4(protected), fps)
    vals = [v["invisibility"] for v in per_clip.values()]
    return {
        "ssim": float(np.mean([v["ssim"] for v in per_clip.values()])),
        "vmaf": None if None in vals else float(np.mean([v["vmaf"] for v in per_clip.values()])),
        "invisibility": None if None in vals else float(np.mean(vals)),
        "per_clip": per_clip,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="my_method")
    ap.add_argument("--config", default=str(REPO_ROOT / "eval" / "config.yaml"))
    args = ap.parse_args()

    cfg = eh.load_yaml(args.config)
    _, data_cfg = eh.load_data_config(REPO_ROOT)
    gen_root = eh.data_root(REPO_ROOT, data_cfg, "generated_root", "data/generated") / args.method
    summaries = sorted(gen_root.rglob("generation_summary.json"))
    if not summaries:
        sys.exit(f"no generation_summary.json under {gen_root}; "
                 f"run `python eval/run.py --method {args.method}` first")

    print(f"[method] {args.method}")
    print("\n== invisibility ==")
    inv = compute_invisibility(data_cfg, args.method)
    vm = "n/a" if inv["vmaf"] is None else f"{inv['vmaf']:.2f}"
    iv = "n/a" if inv["invisibility"] is None else f"{inv['invisibility']:.2f}"
    print(f"  vmaf={vm}  ssim={inv['ssim']:.4f}  invisibility={iv}  (mean over clips)")

    ref = cfg["i2v"]["generation"]["reference_image"]
    original, _ = eh.clip_paths(REPO_ROOT, data_cfg, args.method, ref)
    ref_emb = SC.facenet_embedding(io_video.load_mp4(original)[int(ref.get("frame", 0))])
    if ref_emb is None:
        sys.exit(f"no face found by MTCNN in {original} frame {ref.get('frame', 0)}")

    tracks = {}
    print("\n== tracks ==")
    for summary_path in summaries:
        task, label, model = summary_path.relative_to(gen_root).parts[:3]
        name = f"{task}/{label}/{model}"
        videos = []
        for item in eh.read_json(summary_path).get("generated", []):
            sc = SC.score_video(io_video.load_mp4(REPO_ROOT / item["path"]), item["prompt"], ref_emb)
            sc.pop("per_frame")
            sc.update(video=item["path"], prompt=item["prompt"])
            videos.append(sc)
        if not videos:
            continue
        keys = ("face", "prompt_following", "image_quality", "track")
        mean = {k: float(np.mean([v[k] for v in videos])) for k in keys}
        tracks[name] = {**mean, "videos": videos}
        print(f"  [{name}] face={mean['face']:.2f}  prompt_following={mean['prompt_following']:.2f}  "
              f"image_quality={mean['image_quality']:.2f}  track={mean['track']:.2f}  "
              f"({len(videos)} video(s))")

    out = {"method": args.method, "reference_image": ref, "invisibility": inv, "tracks": tracks}
    out_path = eh.data_root(REPO_ROOT, data_cfg, "results_root", "data/results") / args.method / "results_competition.json"
    eh.write_json(out, out_path)
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
