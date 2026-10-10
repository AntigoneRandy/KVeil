import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_SHA256 = "c4e97066405f2b428a2abddc4186a8ac545f755c84efaf8927bb3cf0330a66c1"
EXPECTED_BYTES = 20286405632


def sha256(path, chunk=1 << 20):
    """Stream the file and return its SHA256 hex digest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(REPO_ROOT / "data" / "original" / "video.tar"))
    args = ap.parse_args()
    path = Path(args.archive)

    print(f"== check {path.name} ==")
    if not path.exists():
        sys.exit(f"{path} not found — place video.tar at data/original/ first")

    size = path.stat().st_size
    if size != EXPECTED_BYTES:
        sys.exit(f"size is {size} bytes, expected {EXPECTED_BYTES} — re-download (gdown may have saved an HTML/partial file)")
    print(f"[size] {size} bytes")

    print("[hash] computing sha256 (takes a few minutes)...")
    digest = sha256(path)
    if digest != EXPECTED_SHA256:
        sys.exit(f"sha256 mismatch — got {digest}, expected {EXPECTED_SHA256}; re-download the official MEAD video.tar")
    print(f"[ok]  {path.name} matches the official checksum")
    print("done.")


if __name__ == "__main__":
    main()
