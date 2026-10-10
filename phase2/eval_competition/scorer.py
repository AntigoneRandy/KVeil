from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

YUNET_ONNX = os.environ.get(
    "YUNET_ONNX",
    str(Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"))
SIGLIP2_ID = os.environ.get("SIGLIP2_ID", "google/siglip2-so400m-patch16-384")
VMAF_BIN = os.environ.get("VMAF_BIN", "vmaf")
VMAF_MODEL = os.environ.get("VMAF_MODEL", "version=vmaf_v0.6.1")
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
PF_CAP = 0.2
FACE_CAP = 100.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_mtcnn = _facenet = _yunet = _musiq = _siglip = _siglip_proc = _ssim = None


def _get_face():
    global _mtcnn, _facenet
    if _mtcnn is None:
        from facenet_pytorch import MTCNN, InceptionResnetV1
        _mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, post_process=True,
                       select_largest=True, device=DEVICE)
        _facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
    return _mtcnn, _facenet


def _get_yunet(w, h):
    global _yunet
    if _yunet is None:
        if not os.path.isfile(YUNET_ONNX):
            raise FileNotFoundError(f"YuNet model not found at {YUNET_ONNX}")
        _yunet = cv2.FaceDetectorYN.create(YUNET_ONNX, "", (w, h))
    _yunet.setInputSize((w, h))
    return _yunet


def _get_musiq():
    global _musiq
    if _musiq is None:
        import pyiqa
        _musiq = pyiqa.create_metric("musiq", device=DEVICE)
    return _musiq


def _get_siglip():
    global _siglip, _siglip_proc
    if _siglip is None:
        from transformers import AutoModel, AutoProcessor
        _siglip_proc = AutoProcessor.from_pretrained(SIGLIP2_ID)
        _siglip = AutoModel.from_pretrained(
            SIGLIP2_ID, dtype=torch.float16).eval().to(DEVICE)
    return _siglip, _siglip_proc


def facenet_embedding(frame_rgb_u8):
    from PIL import Image
    mtcnn, facenet = _get_face()
    aligned = mtcnn(Image.fromarray(np.ascontiguousarray(frame_rgb_u8)))
    if aligned is None:
        return None
    with torch.no_grad():
        emb = facenet(aligned.unsqueeze(0).to(DEVICE))
    return emb[0].float().cpu().numpy()


def yunet_has_face(frame_rgb_u8) -> bool:
    h, w = frame_rgb_u8.shape[:2]
    det = _get_yunet(w, h)
    _rc, faces = det.detect(np.ascontiguousarray(frame_rgb_u8[..., ::-1]))
    return faces is not None and len(faces) > 0


def face_scores(video_u8, ref_emb):
    if ref_emb is None:
        return None
    per_frame, uncapped, mt_ok, yn_ok, coss = [], [], [], [], []
    for t in range(video_u8.shape[0]):
        f = video_u8[t]
        emb = facenet_embedding(f)
        y = yunet_has_face(f)
        mt_ok.append(emb is not None)
        yn_ok.append(bool(y))
        if emb is None or not y:
            per_frame.append(100.0)
            uncapped.append(100.0)
            coss.append(None)
        else:
            c = float(np.dot(emb, ref_emb))
            coss.append(c)
            raw = 100.0 * max(0.0, 1.0 - c)
            uncapped.append(raw)
            per_frame.append(min(FACE_CAP, raw))
    return dict(per_frame=per_frame, per_frame_uncapped=uncapped,
                mtcnn_ok=mt_ok, yunet_ok=yn_ok, cos=coss)


@torch.no_grad()
def musiq_per_frame(video_u8, batch=8):
    m = _get_musiq()
    vals = []
    for i in range(0, video_u8.shape[0], batch):
        x = torch.from_numpy(np.ascontiguousarray(video_u8[i:i + batch])).to(DEVICE)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        vals += [float(v) for v in m(x).flatten().cpu()]
    return vals


@torch.no_grad()
def siglip_cos_per_frame(video_u8, prompt, batch=16):
    model, proc = _get_siglip()
    tin = proc(text=[prompt], padding="max_length", truncation=True, return_tensors="pt")
    tin = {k: v.to(DEVICE) for k, v in tin.items()}
    temb = model.get_text_features(**tin).float()
    temb = temb / temb.norm(dim=-1, keepdim=True)

    out = []
    for i in range(0, video_u8.shape[0], batch):
        imgs = [video_u8[j] for j in range(i, min(i + batch, video_u8.shape[0]))]
        iin = proc(images=imgs, return_tensors="pt")
        iin = {k: (v.to(DEVICE).half() if v.dtype == torch.float32 else v.to(DEVICE))
               for k, v in iin.items()}
        iemb = model.get_image_features(**iin).float()
        iemb = iemb / iemb.norm(dim=-1, keepdim=True)
        out += [float(v) for v in (iemb @ temb.T).flatten().cpu()]
    return out


def score_video(video_u8, prompt, ref_emb):
    n = int(video_u8.shape[0])
    fs = face_scores(video_u8, ref_emb)
    musiq = musiq_per_frame(video_u8)
    cos = siglip_cos_per_frame(video_u8, prompt)

    iq_pf = [float(np.clip(100.0 - m, 0.0, 100.0)) for m in musiq]
    pf_pf = [100.0 * (PF_CAP - float(np.clip(c, 0.0, PF_CAP))) / PF_CAP for c in cos]

    face = float(np.mean(fs["per_frame"])) if fs else None
    face_uncapped = float(np.mean(fs["per_frame_uncapped"])) if fs else None
    iq = float(np.mean(iq_pf))
    pf = float(np.mean(pf_pf))
    track = float(np.mean([face, pf, iq])) if face is not None else None

    det_fail = (float(np.mean([not (a and b) for a, b in zip(fs["mtcnn_ok"], fs["yunet_ok"])]))
                if fs else None)
    return dict(
        n_frames=n, face=face, face_uncapped=face_uncapped,
        image_quality=iq, prompt_following=pf, track=track,
        raw=dict(musiq_mean=float(np.mean(musiq)), siglip_cos_mean=float(np.mean(cos)),
                 detect_fail_rate=det_fail,
                 mtcnn_fail_rate=float(np.mean([not a for a in fs["mtcnn_ok"]])) if fs else None,
                 yunet_fail_rate=float(np.mean([not b for b in fs["yunet_ok"]])) if fs else None,
                 facenet_cos_mean_when_detected=(
                     float(np.mean([c for c in fs["cos"] if c is not None]))
                     if fs and any(c is not None for c in fs["cos"]) else None)),
        alt=dict(
            image_quality=float(np.clip(100.0 - np.mean(musiq), 0.0, 100.0)),
            prompt_following=100.0 * (PF_CAP - float(np.clip(np.mean(cos), 0.0, PF_CAP))) / PF_CAP),
        per_frame=dict(face=fs["per_frame"] if fs else None,
                       face_uncapped=fs["per_frame_uncapped"] if fs else None,
                       musiq=musiq, siglip_cos=cos,
                       mtcnn_ok=fs["mtcnn_ok"] if fs else None,
                       yunet_ok=fs["yunet_ok"] if fs else None,
                       facenet_cos=fs["cos"] if fs else None),
    )


@torch.no_grad()
def ssim(orig_u8, prot_u8, batch=16):
    global _ssim
    if _ssim is None:
        import pyiqa
        _ssim = pyiqa.create_metric("ssim", device=DEVICE)
    vals = []
    for i in range(0, orig_u8.shape[0], batch):
        a = torch.from_numpy(np.ascontiguousarray(orig_u8[i:i + batch])).to(DEVICE)
        b = torch.from_numpy(np.ascontiguousarray(prot_u8[i:i + batch])).to(DEVICE)
        a = a.permute(0, 3, 1, 2).float() / 255.0
        b = b.permute(0, 3, 1, 2).float() / 255.0
        vals += [float(v) for v in _ssim(b, a).flatten().cpu()]
    return float(np.mean(vals))


def _to_y4m(frames_u8, path, fps):
    t, h, w, _ = frames_u8.shape
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, input=np.ascontiguousarray(frames_u8).tobytes(), check=True)


def vmaf(orig_u8, prot_u8, fps=24):
    if not (Path(VMAF_BIN).exists() or shutil.which(VMAF_BIN)):
        warnings.warn(f"vmaf: binary '{VMAF_BIN}' not found; set VMAF_BIN")
        return None
    tmp = Path(tempfile.mkdtemp(prefix="vmaf_"))
    try:
        ref, dist, out = tmp / "ref.y4m", tmp / "dist.y4m", tmp / "out.json"
        _to_y4m(orig_u8, ref, fps)
        _to_y4m(prot_u8, dist, fps)
        p = subprocess.run([VMAF_BIN, "-r", str(ref), "-d", str(dist), "-m", VMAF_MODEL,
                            "--json", "-o", str(out)], capture_output=True, text=True)
        if p.returncode != 0 or not out.exists():
            warnings.warn(f"vmaf: CLI failed rc={p.returncode}: {p.stderr[-300:]}")
            return None
        return float(json.load(open(out))["pooled_metrics"]["vmaf"]["mean"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def invisibility(orig_u8, prot_u8, fps=24):
    s = ssim(orig_u8, prot_u8)
    v = vmaf(orig_u8, prot_u8, fps)
    return dict(ssim=s, vmaf=v,
                invisibility=None if v is None else float((v + 100.0 * s) / 2.0))

