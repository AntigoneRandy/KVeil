import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils import io_video, i2v, metrics       # noqa: E402


def attack_avg3(video):
    """Each frame -> mean of its 3-frame window; edges use a forward/backward window."""
    v = video.astype(np.int16)
    out = video.copy()
    out[1:-1] = ((v[:-2] + v[1:-1] + v[2:]) // 3).astype(np.uint8)
    out[0] = ((v[0] + v[1] + v[2]) // 3).astype(np.uint8)
    out[-1] = ((v[-3] + v[-2] + v[-1]) // 3).astype(np.uint8)
    return out


ATTACKS = {
    "avg3": attack_avg3,
}


def save_generated_mp4(frames_uint8, path, fps=30):
    from diffusers.utils import export_to_video

    pil = [Image.fromarray(f) for f in frames_uint8]
    path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(pil, str(path), fps=fps)


def run_i2v_pass(pipe, clip_uint8, frame_idx, prompt, seed, i2v_cfg, ref_emb, out_mp4):
    ref_frame = clip_uint8[frame_idx]
    generated = i2v.run_i2v(
        pipe, ref_frame, prompt,
        seed=seed,
        num_frames=i2v_cfg["num_frames"],
        num_inference_steps=i2v_cfg["num_inference_steps"],
        guidance_scale=i2v_cfg["guidance_scale"],
        negative_prompt=i2v_cfg["negative_prompt"],
    )
    fd_v = metrics.face_detection(generated)
    fs_v = metrics.face_similarity(generated, ref_emb) if ref_emb is not None else None
    pf_v = metrics.prompt_following(generated, prompt)
    brisque_v = metrics.brisque(generated)
    save_generated_mp4(generated, out_mp4)
    return {"face_detection": fd_v, "face_similarity": fs_v, "prompt_following": pf_v, "brisque": brisque_v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="my_method",
                    help="evaluate data/protected/<method>/ submission, any method name "
                         "with a correctly formatted output (incl. protect/external/)")
    ap.add_argument("--config", default=str(REPO_ROOT / "eval" / "config.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ident, vid, frame_idx = cfg["clip"]["id"], cfg["clip"]["vid"], cfg["clip"]["frame"]
    prompt = cfg["prompt"]
    seed = cfg["seed"]
    tracks_cfg = cfg["tracks"]
    i2v_cfg = cfg["i2v"]
    budget_cfg = cfg["budget"]

    mp4_path = REPO_ROOT / "data" / "preprocessed" / ident / f"{vid}.mp4"
    prot_path = REPO_ROOT / "data" / "protected" / args.method / "submission" / ident / f"{vid}.mp4"
    if not mp4_path.exists():
        sys.exit(f"missing {mp4_path} — run data_preparation/preprocess.py first")
    if not prot_path.exists():
        sys.exit(f"missing {prot_path} — run `python protect/run.py --method {args.method}` first")

    print(f"[clip] {ident}/{vid}   method={args.method}")
    orig = io_video.load_mp4(mp4_path)
    prot = io_video.load_mp4(prot_path)

    # invisibility constraint
    print("\n== invisibility ==")
    eps = int(np.abs(orig.astype(np.int16) - prot.astype(np.int16)).max())
    within_budget = bool(eps <= budget_cfg["linf"])
    psnr_v = metrics.psnr(orig, prot)
    ref_emb, _ = metrics.face_embedding(orig[frame_idx])
    print(
        f"  eps={eps}  psnr={psnr_v:.2f}  within_budget={within_budget}  "
        f"ref_face={'yes' if ref_emb is not None else 'NO'}"
    )
    del orig
    torch.cuda.empty_cache(); gc.collect()

    out_dir = REPO_ROOT / "data" / "generated" / args.method / ident / vid

    eff_enabled = tracks_cfg.get("effectiveness", {}).get("enabled", False)
    rob_cfg = tracks_cfg.get("robustness", {})
    rob_enabled = rob_cfg.get("enabled", False)
    rob_attacks = rob_cfg.get("attacks", []) if rob_enabled else []

    if eff_enabled or rob_attacks:
        print("\n[load] Wan2.2-TI2V-5B ...")
        pipe = i2v.load_wan_i2v()
        print("[load] pipeline ready")

    eff_scores = None
    if eff_enabled:
        print(f"\n== effectiveness ==")
        eff_scores = run_i2v_pass(
            pipe, prot, frame_idx, prompt, seed, i2v_cfg, ref_emb,
            out_dir / "effectiveness.mp4",
        )
        fs_str = "None" if eff_scores["face_similarity"] is None else f"{eff_scores['face_similarity']:.3f}"
        print(f"  face_detection={eff_scores['face_detection']:.3f}  face_similarity={fs_str}  prompt_following={eff_scores['prompt_following']:.3f}  brisque={eff_scores['brisque']:.1f}")

    rob_scores = {}
    if rob_attacks:
        print("\n== robustness ==")
        for atk in rob_attacks:
            if atk not in ATTACKS:
                sys.exit(f"unknown attack: {atk}")
            attacked = ATTACKS[atk](prot)
            scores = run_i2v_pass(
                pipe, attacked, frame_idx, prompt, seed, i2v_cfg, ref_emb,
                out_dir / "robustness" / f"{atk}.mp4",
            )
            fs_str = "None" if scores["face_similarity"] is None else f"{scores['face_similarity']:.3f}"
            print(f"  [{atk}] face_detection={scores['face_detection']:.3f}  face_similarity={fs_str}  prompt_following={scores['prompt_following']:.3f}  brisque={scores['brisque']:.1f}")
            rob_scores[atk] = scores

    scores_out = {
        "method": args.method,
        "config": {
            "id": ident, "vid": vid, "frame": frame_idx,
            "num_frames": i2v_cfg["num_frames"],
            "num_inference_steps": i2v_cfg["num_inference_steps"],
            "guidance_scale": i2v_cfg["guidance_scale"],
            "prompt": prompt,
            "seed": seed,
            "budget_linf": budget_cfg["linf"],
        },
        "invisibility": {
            "eps": eps,
            "psnr": psnr_v,
            "within_budget": within_budget,
        },
    }
    if eff_scores is not None:
        scores_out["effectiveness"] = eff_scores
    if rob_scores:
        scores_out["robustness"] = rob_scores

    out_path = REPO_ROOT / "data" / f"results_{args.method}.json"
    out_path.write_text(json.dumps(scores_out, indent=2, default=str))
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
