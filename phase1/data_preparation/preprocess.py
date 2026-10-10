import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _probe_wh(path):
    """Return (width, height) of the first video stream."""
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x", str(path),
    ]).decode().strip()
    w, h = out.split("x")
    return int(w), int(h)


def _center_crop_43(w, h):
    """Center crop box (cw, ch, cx, cy) cropping (w, h) to 4:3."""
    cw = min(w, h * 4 // 3)
    ch = min(h, w * 3 // 4)
    return cw, ch, (w - cw) // 2, (h - ch) // 2


def _is_valid(path, target):
    """True if path matches target resolution and exact frame count."""
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames,width,height",
        "-of", "default=nw=1", str(path),
    ]).decode()
    f = dict(l.split("=", 1) for l in out.strip().splitlines() if "=" in l)
    return (
        f.get("nb_read_frames") == str(target["num_frames"])
        and f.get("width") == str(target["width"])
        and f.get("height") == str(target["height"])
    )


def _tar_prefix(tar):
    """Top-level directory inside the archive (e.g. 'video')."""
    names = tar.getnames()
    return names[0].split("/", 1)[0] if names else ""


def extract_clip(tar, prefix, src_rel, dst):
    """Stream one clip out of the archive to dst if not already present."""
    if dst.exists():
        return
    member = f"{prefix}/{src_rel}" if prefix else src_rel
    try:
        fsrc = tar.extractfile(member)
    except KeyError:
        sys.exit(f"error: {member} not found in archive; check config clip paths")
    if fsrc is None:
        sys.exit(f"error: {member} is not a regular file in archive")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with fsrc, open(dst, "wb") as fdst:
        shutil.copyfileobj(fsrc, fdst)
    print(f"[extract] {member} -> {dst.relative_to(REPO_ROOT)}")


def process_clip(src, out_path, target):
    """Center-crop to 4:3, scale, pad/truncate to num_frames, libx264 30fps encode."""
    if out_path.exists() and _is_valid(out_path, target):
        print(f"[skip] {out_path.relative_to(REPO_ROOT)} already valid")
        return
    w, h = _probe_wh(src)
    cw, ch, cx, cy = _center_crop_43(w, h)
    tw, th, nf = target["width"], target["height"], target["num_frames"]
    vf = (
        f"crop={cw}:{ch}:{cx}:{cy},"
        f"scale={tw}:{th}:flags=lanczos,"
        f"tpad=stop=8:stop_mode=clone"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", vf,
        "-frames:v", str(nf),
        "-r", "30",
        "-an",
        "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p",
        str(out_path),
    ], check=True)
    if not _is_valid(out_path, target):
        sys.exit(f"error: produced clip failed validation: {out_path}")
    print(f"[ok]  {out_path.relative_to(REPO_ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", default=str(REPO_ROOT / "data_preparation" / "config.yaml")
    )
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    src_root = (REPO_ROOT / cfg["source_root"]).resolve()
    out_dir = (REPO_ROOT / cfg["output_dir"]).resolve()
    archive = (REPO_ROOT / cfg["archive"]).resolve()
    target = cfg["target"]

    total = sum(len(i["clips"]) for i in cfg["identities"])
    print(f"== preprocess {total} video clip(s) across {len(cfg['identities'])} identity(ies) -> {out_dir} ==")

    tar = None
    prefix = ""
    try:
        for ident in cfg["identities"]:
            for clip in ident["clips"]:
                out_path = out_dir / ident["id"] / f"{clip['name']}.mp4"
                if out_path.exists() and _is_valid(out_path, target):
                    print(f"[skip] {out_path.relative_to(REPO_ROOT)} already valid")
                    continue
                src_path = src_root / ident["id"] / clip["src"]
                if not src_path.exists():
                    if tar is None:
                        if not archive.exists():
                            sys.exit(
                                f"error: neither {src_path} nor archive {archive} found; "
                                "place video.tar and run check_archive.py first"
                            )
                        tar = tarfile.open(archive)
                        prefix = _tar_prefix(tar)
                    extract_clip(tar, prefix, clip["src"], src_path)
                process_clip(src_path, out_path, target)
    finally:
        if tar is not None:
            tar.close()
    print("done.")


if __name__ == "__main__":
    main()
