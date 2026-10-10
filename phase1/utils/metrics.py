import contextlib
import io
import warnings

import numpy as np
import onnxruntime as ort
import torch

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
ort.set_default_logger_severity(3)

_face_app = None


def _silent():
    return contextlib.redirect_stdout(io.StringIO())


# Face model: 
# InsightFace buffalo_l = SCRFD (Guo et al., ICLR 2022)
# ArcFace (Deng et al., CVPR 2019)
def _face_analysis():
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        with _silent():
            _face_app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app


def face_embedding(frame_uint8):
    app = _face_analysis()
    faces = app.get(frame_uint8[..., ::-1])  # InsightFace expects BGR
    if not faces:
        return None, 0.0
    faces.sort(
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )
    f = faces[0]
    return f.normed_embedding, float(f.det_score)


# fdfr/ism metric definitions follow Anti-DreamBooth (Van Le et al., ICCV 2023).
def face_detection(video_uint8):
    n_fail = sum(
        face_embedding(video_uint8[t])[0] is None
        for t in range(video_uint8.shape[0])
    )
    return n_fail / video_uint8.shape[0]


def face_similarity(video_uint8, ref_embedding):
    sims = []
    for t in range(video_uint8.shape[0]):
        emb, _ = face_embedding(video_uint8[t])
        if emb is not None:
            sims.append(float(np.dot(emb, ref_embedding)))
    return float(np.mean(sims)) if sims else None


# PSNR: peak signal-to-noise ratio of the protected clip vs. the original.
def psnr(orig_uint8, prot_uint8):
    mse = np.mean((orig_uint8.astype(np.float64) - prot_uint8.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(20.0 * np.log10(255.0 / np.sqrt(mse)))


# Frame-averaged CLIP image-text cosine (CLIPScore / CLIP-SIM), the public-eval metric.
# Model: openai/clip-vit-base-patch32 (CLIP, Radford et al. 2021), via HF transformers.
_clip_model = None
_clip_processor = None
_CLIP_NAME = "openai/clip-vit-base-patch32"


def _clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        with _silent():
            _clip_model = CLIPModel.from_pretrained(_CLIP_NAME).cuda().eval()
            _clip_processor = CLIPProcessor.from_pretrained(_CLIP_NAME)
    return _clip_model, _clip_processor


def prompt_following(video_uint8, prompt):
    """Mean per-frame image-text cosine over all frames (prompt-following strength)."""
    from PIL import Image
    model, proc = _clip()
    frames = [Image.fromarray(f) for f in video_uint8]
    inputs = proc(text=[prompt], images=frames, return_tensors="pt", padding=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return float((img @ txt.T).squeeze(-1).mean())


_brisque = None


# BRISQUE no-reference image quality (no-ref NSS, via pyiqa / IQA-PyTorch).
def brisque(video_uint8):
    """Per-frame BRISQUE of the generated video, averaged. Higher = more degraded = stronger protection."""
    global _brisque
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if _brisque is None:
        import pyiqa
        _brisque = pyiqa.create_metric("brisque", device=dev)
    out = []
    for i in range(0, len(video_uint8), 16):
        t = torch.from_numpy(video_uint8[i:i + 16].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(dev)
        with torch.no_grad():
            out.append(_brisque(t).flatten().cpu())
    return float(torch.cat(out).mean())
