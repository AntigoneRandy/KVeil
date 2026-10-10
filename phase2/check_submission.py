import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from utils import io_video                 # noqa: E402


def expected_clips(cfg):
    "Set of '<id>/<name>.mp4' the archive must contain, from the data config."
    out = set()
    for ident in cfg["identities"]:
        for clip in ident["clips"]:
            out.add(f"{ident['id']}/{clip['name']}.mp4")
    return out


def probe(path):
    "ffprobe one clip into the fields we validate, or None if undecodable."
    cmd = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames",
        "-of", "json", str(path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    streams = json.loads(p.stdout or "{}").get("streams", [])
    if not streams:
        return None
    s = streams[0]
    num, den = (s.get("r_frame_rate") or "0/1").split("/")
    s["fps"] = round(int(num) / int(den)) if int(den) else 0
    s["frames"] = int(s.get("nb_read_frames") or 0)
    return s


def check_clip(s):
    "Return a list of failures for one probed clip vs the lossless schema."
    t, h, w, _ = io_video.SCHEMA["shape"]
    want = io_video.LOSSLESS_PROBE
    bad = []
    if s["codec_name"] != want["codec_name"]:
        bad.append(f"codec {s['codec_name']} != {want['codec_name']}")
    if s["pix_fmt"] != want["pix_fmt"]:
        bad.append(f"pix_fmt {s['pix_fmt']} != {want['pix_fmt']} (not the lossless encoding)")
    if (s["width"], s["height"]) != (w, h):
        bad.append(f"size {s['width']}x{s['height']} != {w}x{h}")
    if s["fps"] != io_video.SCHEMA["fps"]:
        bad.append(f"fps {s['fps']} != {io_video.SCHEMA['fps']}")
    if s["frames"] != t:
        bad.append(f"frames {s['frames']} != {t}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(REPO_ROOT / "submission.zip"))
    ap.add_argument("--config", default=str(REPO_ROOT / "data_preparation" / "config.yaml"))
    args = ap.parse_args()

    archive = Path(args.archive)
    print(f"== check {archive.name} ==")
    if not archive.exists():
        sys.exit(f"{archive} not found")
    if not zipfile.is_zipfile(archive):
        sys.exit(f"{archive} is not a valid zip")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    expected = expected_clips(cfg)
    ids = {i["id"] for i in cfg["identities"]}

    zf = zipfile.ZipFile(archive)
    members = [n for n in zf.namelist() if not n.endswith("/")]
    cruft = [n for n in members
             if n.startswith("__MACOSX/") or Path(n).name == ".DS_Store" or Path(n).name.startswith("._")]
    present = set(members) - set(cruft)

    ok = True
    missing = sorted(expected - present)
    extra = sorted(present - expected)

    if cruft:
        ok = False
        print("[structure] FAIL: archive has OS metadata, re-zip with the command in the README")
        for n in cruft[:5]:
            print(f"            - {n}")
    if missing:
        ok = False
        print("[structure] FAIL: missing expected clips")
        for n in missing:
            print(f"            - {n}")
    if extra:
        ok = False
        hint = "  (extra wrapping folder? zip from inside submission/)" if {n.split("/")[0] for n in extra} - ids else ""
        print(f"[structure] FAIL: unexpected files{hint}")
        for n in extra:
            print(f"            - {n}")
    if not (cruft or missing or extra):
        print(f"[structure] ok")

    tmp = Path(tempfile.mkdtemp())
    for rel in sorted(expected & present):
        zf.extract(rel, tmp)
        s = probe(tmp / rel)
        if s is None:
            ok = False
            print(f"[{rel}] FAIL: no decodable video stream")
            continue
        bad = check_clip(s)
        if bad:
            ok = False
            print(f"[{rel}] FAIL: " + ", ".join(bad))
        else:
            print(f"[{rel}] PASS  {s['codec_name']}/{s['pix_fmt']} {s['width']}x{s['height']} {s['fps']}fps {s['frames']}f")

    print()
    if ok:
        print("RESULT: PASS, archive is ready to upload")
    else:
        sys.exit("RESULT: FAIL, fix the issues above before uploading or the submission may be rejected")


if __name__ == "__main__":
    main()
