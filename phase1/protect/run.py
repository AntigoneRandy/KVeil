import argparse
import importlib
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils import io_video                 # noqa: E402


def main():
    methods_dir = Path(__file__).parent / "methods"
    available = sorted(p.stem for p in methods_dir.glob("*.py"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="my_method", choices=available,
                    help="protection method to run (file under protect/methods/)")
    args = ap.parse_args()

    mod = importlib.import_module(f"protect.methods.{args.method}")
    protect = mod.protect

    cfg = yaml.safe_load(open(REPO_ROOT / "data_preparation" / "config.yaml"))
    for i, ident in enumerate(cfg["identities"]):
        ident_id = ident["id"]
        in_dir = REPO_ROOT / "data" / "preprocessed" / ident_id
        sub_dir = REPO_ROOT / "data" / "protected" / args.method / "submission" / ident_id
        prev_dir = REPO_ROOT / "data" / "protected" / args.method / "preview" / ident_id
        files = sorted(in_dir.glob("*.mp4"))
        if not files:
            sys.exit(f"no .mp4 under {in_dir} — run download_and_preprocess.py first")
        if i > 0:
            print()
        print(f"== protect {ident_id} via {args.method} ({len(files)} clips) ==")
        videos = {p.stem: io_video.load_mp4(p) for p in files}
        protected = protect(videos)
        for name, v in protected.items():
            sub_mp4 = sub_dir / f"{name}.mp4"
            prev_mp4 = prev_dir / f"{name}.mp4"
            io_video.save_mp4(v, sub_mp4, lossless=True)    # pixel-exact RGB, submit this
            io_video.save_mp4(v, prev_mp4, lossless=False)  # for viewing
            print(f"[ok]  {sub_mp4.relative_to(REPO_ROOT)}")
            print(f"[ok]  {prev_mp4.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
