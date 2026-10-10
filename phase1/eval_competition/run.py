import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import io_video
import scorer as SC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="my_method")
    ap.add_argument("--config", default=str(REPO_ROOT / "eval" / "config.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ident, vid, frame_idx = cfg["clip"]["id"], cfg["clip"]["vid"], cfg["clip"]["frame"]
    prompt = cfg["prompt"]

    orig_path = REPO_ROOT / "data" / "preprocessed" / ident / f"{vid}.mp4"
    prot_path = REPO_ROOT / "data" / "protected" / args.method / "submission" / ident / f"{vid}.mp4"
    gen_dir = REPO_ROOT / "data" / "generated" / args.method / ident / vid
    for p in (orig_path, prot_path):
        if not p.exists():
            sys.exit(f"missing {p}")

    videos = {}
    if (gen_dir / "effectiveness.mp4").exists():
        videos["effectiveness"] = gen_dir / "effectiveness.mp4"
    for p in sorted((gen_dir / "robustness").glob("*.mp4")):
        videos[f"robustness_{p.stem}"] = p
    if not videos:
        sys.exit(f"no generated videos under {gen_dir}; run `python eval/run.py --method {args.method}` first")

    orig = io_video.load_mp4(orig_path)
    prot = io_video.load_mp4(prot_path)

    print(f"[clip] {ident}/{vid}   method={args.method}")
    print("\n== invisibility ==")
    inv = SC.invisibility(orig, prot)
    vm = "n/a" if inv["vmaf"] is None else f"{inv['vmaf']:.2f}"
    iv = "n/a" if inv["invisibility"] is None else f"{inv['invisibility']:.2f}"
    print(f"  vmaf={vm}  ssim={inv['ssim']:.4f}  invisibility={iv}")

    ref_emb = SC.facenet_embedding(orig[frame_idx])
    if ref_emb is None:
        sys.exit(f"no face found by MTCNN in original frame {frame_idx}")
    del orig, prot

    tracks = {}
    print("\n== tracks ==")
    for name, path in videos.items():
        sc = SC.score_video(io_video.load_mp4(path), prompt, ref_emb)
        sc.pop("per_frame")
        sc["video"] = str(path.relative_to(REPO_ROOT))
        tracks[name] = sc
        print(f"  [{name}] face={sc['face']:.2f}  prompt_following={sc['prompt_following']:.2f}  "
              f"image_quality={sc['image_quality']:.2f}  track={sc['track']:.2f}")

    out = {"method": args.method, "prompt": prompt, "reference_frame": frame_idx,
           "invisibility": inv, "tracks": tracks}
    out_path = REPO_ROOT / "data" / f"results_competition_{args.method}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
