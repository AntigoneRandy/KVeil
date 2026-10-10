from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import math
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanAttnProcessor

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.use_deterministic_algorithms(True, warn_only=True)

WAN22_MODEL_PATH = os.environ.get(
    "WAN22_MODEL_PATH",
    os.environ.get("WAN_MODEL_PATH", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"))

ANTELOPEV2_ROOT = os.path.expanduser(os.environ.get("ANTELOPEV2_ROOT", "~/.insightface"))
TARGET_IMAGE = os.environ.get(
    "SYNTH_TARGET", str(Path(__file__).resolve().parent / "assets" / "original.png"))

H, W = 480, 640
EPS_PIXEL = 16
ANCHOR = 0

FACE_DILATE_FRAC = 0.35
FACE_FEATHER_FRAC = 0.06
LAB_GAIN = 20.0
LAB_W_L, LAB_W_AB = 1.2, 1.0
LAB_SIGMA_HI = 6.0
ALIGN_TPS_LAMBDA, ALIGN_N_BOUNDARY = 1e-3, 5
ALIGN_ANTIALIAS_THRESH = 1.5
TRACK_TPS_LAMBDA, TRACK_N_BOUNDARY = 1e-3, 6

NUM_STEPS = 200
EPSILON = 16 / 255 * 2
STEP_SIZE = EPSILON / 20
MOMENTUM_BETA = 0.9
SEED = 0
EOT_LAYERS = [0, 3, 6]
EOT_SAMPLES = 3
MF_NUM = 5
NUM_FRAMES = 17
NUM_INFERENCE_STEPS_ATTACK = 25
T_LOW = T_HIGH = 5
K_LAMBDA = 0.5
VAE_L2 = 1.0
VAE_DSSIM = 1.0
EOT_PROMPTS = ["",
               "a person",
               "the person is talking to the camera",
               "a woman speaking to the camera",
               "a man talking",
               "continue the video",
               "a portrait video of a person"]

S_PER_FRAME = (H // 16 // 2) * (W // 16 // 2)
LATENT_FRAMES = (NUM_FRAMES - 1) // 4 + 1
LATENT_H, LATENT_W = H // 16, W // 16

_PIPE = None
_CAPTURED: dict[int, dict[str, torch.Tensor]] = {}

def _get_pipe():
    global _PIPE
    if _PIPE is None:
        vae = AutoencoderKLWan.from_pretrained(WAN22_MODEL_PATH, subfolder="vae",
                                               torch_dtype=torch.float32)
        pipe = WanImageToVideoPipeline.from_pretrained(WAN22_MODEL_PATH, vae=vae,
                                                       torch_dtype=torch.bfloat16).to("cuda")
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.transformer.enable_gradient_checkpointing()
        for module in (pipe.vae, pipe.transformer, pipe.text_encoder):
            for p in module.parameters():
                p.requires_grad_(False)
        _PIPE = pipe
    return _PIPE

def encode_first_frame_latent(pipe, img_neg1pos1):
    encoded = pipe.vae.encode(img_neg1pos1.unsqueeze(2).to(dtype=pipe.vae.dtype))
    z = encoded.latent_dist.mode().to(pipe.transformer.dtype)
    dt, dev = z.dtype, z.device
    mean = torch.tensor(pipe.vae.config.latents_mean, dtype=dt, device=dev).view(1, -1, 1, 1, 1)
    std_inv = 1.0 / torch.tensor(pipe.vae.config.latents_std, dtype=dt, device=dev).view(1, -1, 1, 1, 1)
    return (z - mean) * std_inv

def compute_temp_ts(first_frame_mask, t):
    return (first_frame_mask[0, 0, :, ::2, ::2] * t).flatten().unsqueeze(0)

class CaptureWanAttnProcessor(WanAttnProcessor):

    def __init__(self, block_id: int):
        super().__init__()
        self.block_id = block_id

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query_h = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key_h = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        value_h = value.unflatten(2, (attn.heads, -1))

        def apply_rotary_emb(hs, freqs_cos, freqs_sin):
            x1, x2 = hs.unflatten(-1, (-1, 2)).unbind(-1)
            cos = freqs_cos[..., 0::2]
            sin = freqs_sin[..., 1::2]
            out = torch.empty_like(hs)
            out[..., 0::2] = x1 * cos - x2 * sin
            out[..., 1::2] = x1 * sin + x2 * cos
            return out.type_as(hs)

        query_h = apply_rotary_emb(query_h, *rotary_emb)
        key_h = apply_rotary_emb(key_h, *rotary_emb)

        _CAPTURED[self.block_id] = {"q": query_h.flatten(2, 3),
                                    "k": key_h.flatten(2, 3),
                                    "v": value}

        from diffusers.models.attention_dispatch import dispatch_attention_fn
        out = dispatch_attention_fn(query_h, key_h, value_h, attn_mask=attention_mask,
                                    dropout_p=0.0, is_causal=False,
                                    backend=self._attention_backend)
        out = out.flatten(2, 3).type_as(query_h)
        return attn.to_out[1](attn.to_out[0](out))

def install_capture_patches(transformer, block_ids):
    _CAPTURED.clear()
    for bid in block_ids:
        transformer.blocks[bid].attn1.set_processor(CaptureWanAttnProcessor(bid))

def partial_dit_forward(transformer, hidden_states, timestep, encoder_hidden_states, target_block):
    rotary_emb = transformer.rope(hidden_states)
    hidden_states = transformer.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
    ts_seq_len = timestep.shape[1]
    _, timestep_proj, ehs_emb, _ = transformer.condition_embedder(
        timestep.flatten(), encoder_hidden_states, None, timestep_seq_len=ts_seq_len)
    timestep_proj = timestep_proj.unflatten(2, (6, -1))
    for i in range(min(target_block, len(transformer.blocks) - 1) + 1):
        hidden_states = transformer._gradient_checkpointing_func(
            transformer.blocks[i], hidden_states, ehs_emb, timestep_proj, rotary_emb)
    return hidden_states

def encode_prompt(pipe, prompt):
    pos, _ = pipe.encode_prompt(prompt=prompt, do_classifier_free_guidance=False,
                                num_videos_per_prompt=1, device=pipe.device,
                                dtype=pipe.transformer.dtype)
    return pos

def preprocess_img(pil):
    arr = np.asarray(pil.convert("RGB").resize((W, H))).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1

_FFMPEG = None

def _ffmpeg_bin():
    global _FFMPEG
    if _FFMPEG is None:
        cands = []
        if os.environ.get("PHASE2_FFMPEG"):
            cands.append(os.environ["PHASE2_FFMPEG"])
        if shutil.which("ffmpeg"):
            cands.append(shutil.which("ffmpeg"))
        for c in cands:
            if c and os.path.isfile(c) and os.access(c, os.X_OK):
                _FFMPEG = c
                return _FFMPEG
        raise RuntimeError(f"no usable ffmpeg; set PHASE2_FFMPEG. tried: {cands}")
    return _FFMPEG

def _h264_frame(img_hwc_u8, crf):
    ff = _ffmpeg_bin()
    h, w = img_hwc_u8.shape[:2]
    with tempfile.TemporaryDirectory() as d:
        raw, mp4, out = os.path.join(d, "i.raw"), os.path.join(d, "c.mp4"), os.path.join(d, "o.raw")
        img_hwc_u8.tofile(raw)
        subprocess.run([ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                        "-r", "25", "-i", raw, "-c:v", "libx264", "-crf", str(int(crf)),
                        "-pix_fmt", "yuv420p", "-frames:v", "1", mp4],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        subprocess.run([ff, "-y", "-i", mp4, "-f", "rawvideo", "-pix_fmt", "rgb24", out],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        comp = np.fromfile(out, dtype=np.uint8)
    return comp[:h * w * 3].reshape(h, w, 3) if comp.size >= h * w * 3 else img_hwc_u8

class RealH264:

    def __init__(self, crf_range=(32, 35)):
        self.crf_range = crf_range

    def __call__(self, x01):
        outs = []
        for b in range(x01.shape[0]):
            img = (x01[b].detach().permute(1, 2, 0).clamp(0, 1).float().cpu().numpy() * 255
                   ).round().astype(np.uint8)
            crf = int(np.random.randint(self.crf_range[0], self.crf_range[1] + 1))
            comp = _h264_frame(img, crf)
            outs.append(torch.from_numpy(np.ascontiguousarray(comp)).permute(2, 0, 1).float() / 255.0)
        comp_t = torch.stack(outs).to(x01.device, x01.dtype)
        return x01 + (comp_t - x01).detach()

class RandomDownsampleUpsample(torch.nn.Module):

    def __init__(self, scale_range=(0.3, 0.7), modes=("bilinear", "bicubic", "area")):
        super().__init__()
        self.scale_range = scale_range
        self.modes = modes

    def forward(self, x):
        h, w = x.shape[-2:]
        s = random.uniform(*self.scale_range)
        m_down, m_up = random.choice(self.modes), random.choice(self.modes)
        x_low = F.interpolate(x, size=(max(8, int(h * s)), max(8, int(w * s))), mode=m_down,
                              align_corners=False if m_down != "area" else None)
        return F.interpolate(x_low, size=(h, w), mode=m_up,
                             align_corners=False if m_up != "area" else None).clamp(0.0, 1.0)

def build_eot_augs(device="cuda"):
    import kornia.augmentation as KA
    return [
        ("identity",      None),
        ("h264",          RealH264(crf_range=(32, 35))),
        ("jpeg",          KA.RandomJPEG(jpeg_quality=(35.0, 70.0), p=1.0).to(device)),
        ("blur",          KA.RandomGaussianBlur(kernel_size=(7, 7), sigma=(1.0, 3.0), p=1.0).to(device)),
        ("noise",         KA.RandomGaussianNoise(mean=0.0, std=0.05, p=1.0).to(device)),
        ("downsample_up", RandomDownsampleUpsample(scale_range=(0.3, 0.7)).to(device)),
    ]

def apply_eot_aug(I_adv, aug):
    if aug[1] is None:
        return I_adv
    x01 = ((I_adv + 1.0) * 0.5).clamp(0.0, 1.0).to(torch.float32)
    return (aug[1](x01) * 2.0 - 1.0).clamp(-1.0, 1.0).to(I_adv.dtype)

def capacity_ratio_F0(v_full):
    V = v_full[0, :S_PER_FRAME, :]
    centered = V - V.mean(dim=0, keepdim=True)
    return (centered ** 2).sum() / ((V ** 2).sum() + 1e-8)

def k_attract_loss(q_full, k_full, num_heads, head_dim):
    scale = 1.0 / math.sqrt(head_dim)
    q_h = q_full.detach().view(q_full.shape[0], q_full.shape[1], num_heads, head_dim)
    k_h = k_full.view(k_full.shape[0], k_full.shape[1], num_heads, head_dim)
    q_pool = q_h[:, S_PER_FRAME:].mean(dim=1)
    k_pool = k_h[:, :S_PER_FRAME].mean(dim=1)
    return ((q_pool * k_pool).sum(dim=-1) * scale).mean()

def vae_l2_dist(za, zc):
    return ((za.float() - zc.float()) ** 2).mean()

def vae_latent_dssim(za, zc):
    from kornia.losses import ssim_loss
    a, b = za.float(), zc.float()
    B, C, Fr, Hh, Ww = a.shape
    a = a.reshape(B * C * Fr, 1, Hh, Ww)
    b = b.reshape(B * C * Fr, 1, Hh, Ww)
    lo = b.detach().amin(dim=(2, 3), keepdim=True)
    hi = b.detach().amax(dim=(2, 3), keepdim=True)
    scale = (hi - lo).clamp_min(1e-3)
    return ssim_loss((a - lo) / scale, (b - lo) / scale, window_size=7, max_val=1.0)

def attack_background_delta(pipe, src_pils, frozen_base, update_mask):
    from tqdm import tqdm

    num_heads = pipe.transformer.config.num_attention_heads
    head_dim = pipe.transformer.config.attention_head_dim
    N = len(src_pils)

    eot_augs = build_eot_augs("cuda")
    install_capture_patches(pipe.transformer, EOT_LAYERS)

    with torch.no_grad():
        prompt_embs = [encode_prompt(pipe, p) for p in EOT_PROMPTS]

    noise_gen = torch.Generator(device="cpu").manual_seed(SEED)
    noise = torch.randn((1, 48, LATENT_FRAMES, LATENT_H, LATENT_W), generator=noise_gen,
                        device="cpu", dtype=torch.float32).to(device=pipe.device,
                                                              dtype=pipe.transformer.dtype)
    first_frame_mask = torch.ones(1, 1, LATENT_FRAMES, LATENT_H, LATENT_W,
                                  dtype=pipe.transformer.dtype, device=pipe.device)
    first_frame_mask[:, :, 0] = 0

    I_before = [preprocess_img(p).to(device=pipe.device, dtype=pipe.transformer.dtype)
                for p in src_pils]
    with torch.no_grad():
        z_clean = [encode_first_frame_latent(pipe, ib) for ib in I_before]

    delta = torch.empty_like(I_before[0]).uniform_(-EPSILON / 4, EPSILON / 4)
    delta = delta.detach().requires_grad_(True)
    momentum = torch.zeros_like(delta)

    frozen = torch.clamp(frozen_base.detach().to(delta.device, delta.dtype), -EPSILON, EPSILON)
    umask = update_mask.detach().to(delta.device, delta.dtype)

    def delta_eff(j):
        return frozen[j:j + 1] * (1.0 - umask[j:j + 1]) + delta * umask[j:j + 1]

    pbar = tqdm(range(NUM_STEPS), desc=f"bg-δ ({N}f)")
    for _step in pbar:
        if delta.grad is not None:
            delta.grad.zero_()
        loss_acc = 0.0
        last = (-1, -1)
        for _k in range(EOT_SAMPLES):
            j = random.randrange(N)
            I_adv = torch.clamp(I_before[j] + delta_eff(j), -1.0, 1.0)
            I_adv_aug = apply_eot_aug(I_adv, random.choice(eot_augs))
            z_adv = encode_first_frame_latent(pipe, I_adv_aug)

            block = random.choice(EOT_LAYERS)
            prompt_emb = random.choice(prompt_embs)

            t_idx = random.randint(T_LOW, T_HIGH)

            pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS_ATTACK, device=pipe.device)
            with torch.no_grad():
                cur = noise.clone()
                for i in range(t_idx):
                    tt_i = pipe.scheduler.timesteps[i].to(pipe.device)
                    latent_in = (1 - first_frame_mask) * z_adv + first_frame_mask * cur
                    v_pred = pipe.transformer(hidden_states=latent_in,
                                              timestep=compute_temp_ts(first_frame_mask, tt_i),
                                              encoder_hidden_states=prompt_emb,
                                              return_dict=False)[0]
                    cur = pipe.scheduler.step(v_pred, pipe.scheduler.timesteps[i], cur).prev_sample
            latent_in = (1 - first_frame_mask) * z_adv + first_frame_mask * cur
            temp_ts = compute_temp_ts(first_frame_mask, pipe.scheduler.timesteps[t_idx].to(pipe.device))

            _CAPTURED.clear()
            partial_dit_forward(pipe.transformer, latent_in, temp_ts, prompt_emb, block)
            cell = _CAPTURED[block]

            loss = capacity_ratio_F0(cell["v"].float())
            loss = loss - K_LAMBDA * k_attract_loss(cell["q"].float(), cell["k"].float(),
                                                    num_heads, head_dim)
            loss = loss - VAE_L2 * vae_l2_dist(z_adv, z_clean[j])
            loss = loss - VAE_DSSIM * vae_latent_dssim(z_adv, z_clean[j])

            (loss / EOT_SAMPLES).backward()
            loss_acc += loss.item()
            last = (block, j)

        momentum = MOMENTUM_BETA * momentum + delta.grad.data.sign()
        delta.data = torch.clamp(delta.data - STEP_SIZE * momentum.sign(), -EPSILON, EPSILON)
        pbar.set_postfix(loss=f"{loss_acc / EOT_SAMPLES:.4f}", L=last[0], f=last[1])

    keep = (umask.amax(dim=0, keepdim=True) > 0.5).to(umask.dtype)
    d_px = np.round((delta * keep).detach().float().cpu().numpy()[0].transpose(1, 2, 0) * 127.5)
    return np.clip(d_px, -EPS_PIXEL, EPS_PIXEL).astype(np.int16)

_FACE_APP = None
_MASK_APP = None

def _get_face_app():
    global _FACE_APP
    if _FACE_APP is None:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="antelopev2", root=ANTELOPEV2_ROOT,
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                           allowed_modules=["detection", "landmark_3d_68", "landmark_2d_106"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        assert "recognition" not in app.models, f"recognition head must not be loaded: {sorted(app.models)}"
        _FACE_APP = app
    return _FACE_APP

def _get_mask_app():
    global _MASK_APP
    if _MASK_APP is None:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                           allowed_modules=["detection", "landmark_2d_106"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        assert "recognition" not in app.models, f"recognition head must not be loaded: {sorted(app.models)}"
        _MASK_APP = app
    return _MASK_APP

def _largest_face(app, frame_rgb_u8):
    faces = app.get(np.asarray(frame_rgb_u8)[..., ::-1])
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0]

def build_face_union_mask(clip_u8):
    union = np.zeros((H, W), np.uint8)
    sizes = []
    app = _get_mask_app()
    for fi in range(clip_u8.shape[0]):
        face = _largest_face(app, clip_u8[fi])
        if face is None:
            continue
        x1, y1, x2, y2 = face.bbox
        size = float(np.hypot(x2 - x1, y2 - y1))
        sizes.append(size)
        hull = cv2.convexHull(np.round(face.landmark_2d_106).astype(np.int32))
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, hull, 255)
        d = max(1, int(FACE_DILATE_FRAC * size)) | 1
        union = np.maximum(union, cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))))
    if not sizes:
        raise RuntimeError("no face detected in any frame of the clip; the face layer cannot be built")
    sigma = max(1.0, FACE_FEATHER_FRAC * float(np.mean(sizes)))
    mask = np.clip(cv2.GaussianBlur(union.astype(np.float32), (0, 0), sigma) / 255.0, 0.0, 1.0)
    return mask, len(sizes)

def track_landmarks(clip_u8):
    app = _get_face_app()
    T = int(clip_u8.shape[0])
    lm68 = np.full((T, 68, 2), np.nan, np.float32)
    for t in range(T):
        face = _largest_face(app, clip_u8[t])
        if face is None:
            continue
        l3 = getattr(face, "landmark_3d_68", None)
        if l3 is not None:
            lm68[t] = np.asarray(l3, np.float32)[:, :2]
    return lm68

EYE_L, EYE_R = slice(36, 42), slice(42, 48)

def umeyama(src, dst):
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    n, dim = src.shape
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    var_s = (xs ** 2).sum() / n
    U, S, Vt = np.linalg.svd(xd.T @ xs / n)
    Wm = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Wm[-1, -1] = -1.0
    R = U @ Wm @ Vt
    s = float((S * np.diag(Wm)).sum() / var_s)
    A = np.eye(3, dtype=np.float64)
    A[:2, :2] = s * R
    A[:2, 2] = mu_d - s * R @ mu_s
    return A

def _U_tps(r2):
    return r2 * np.log(np.maximum(r2, 1e-12))

def tps_fit(ctrl_src, ctrl_dst, lam):
    ctrl_src, ctrl_dst = np.asarray(ctrl_src, np.float64), np.asarray(ctrl_dst, np.float64)
    n = ctrl_src.shape[0]
    d2 = ((ctrl_src[:, None, :] - ctrl_src[None, :, :]) ** 2).sum(-1)
    P = np.hstack([np.ones((n, 1)), ctrl_src])
    L = np.zeros((n + 3, n + 3), np.float64)
    L[:n, :n] = _U_tps(d2) + lam * np.eye(n)
    L[:n, n:] = P
    L[n:, :n] = P.T
    Y = np.zeros((n + 3, 2), np.float64)
    Y[:n] = ctrl_dst
    return np.linalg.solve(L, Y)

def tps_apply_np(params, ctrl_src, pts, chunk=100_000):
    n = ctrl_src.shape[0]
    w, a = params[:n], params[n:]
    out = np.empty_like(pts)
    for i in range(0, pts.shape[0], chunk):
        p = pts[i:i + chunk]
        d2 = ((p[:, None, :] - ctrl_src[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = _U_tps(d2) @ w + np.hstack([np.ones((p.shape[0], 1)), p]) @ a
    return out

def tps_apply_torch(params, ctrl_src, pts, device, chunk=307200):
    n = ctrl_src.shape[0]
    w = torch.as_tensor(np.ascontiguousarray(params[:n]), dtype=torch.float64, device=device)
    a = torch.as_tensor(np.ascontiguousarray(params[n:]), dtype=torch.float64, device=device)
    ctrl = torch.as_tensor(np.ascontiguousarray(ctrl_src), dtype=torch.float64, device=device)
    out = torch.empty_like(pts)
    for i in range(0, pts.shape[0], chunk):
        p = pts[i:i + chunk]
        d2 = ((p[:, None, :] - ctrl[None, :, :]) ** 2).sum(-1)
        U = d2 * torch.log(torch.clamp(d2, min=1e-12))
        ones = torch.ones((p.shape[0], 1), dtype=torch.float64, device=device)
        out[i:i + chunk] = U @ w + torch.cat([ones, p], 1) @ a
    return out

def boundary_anchors(n_per_side):
    xs = np.linspace(0, W - 1, n_per_side)
    ys = np.linspace(0, H - 1, n_per_side)
    pts = ([(x, 0.0) for x in xs] + [(x, H - 1.0) for x in xs]
           + [(0.0, y) for y in ys] + [(W - 1.0, y) for y in ys])
    return np.unique(np.asarray(pts, np.float64), axis=0)

def align_target_to_anchor(target_rgb_u8, anchor_frame_u8, device):
    app = _get_face_app()
    gf, gt = _largest_face(app, anchor_frame_u8), _largest_face(app, target_rgb_u8)
    if gf is None or gt is None:
        raise RuntimeError("no face detected on the anchor frame or on the target-identity image; cannot align")
    f_lm = np.asarray(gf.landmark_3d_68, np.float64)[:, :2]
    t_lm = np.asarray(gt.landmark_3d_68, np.float64)[:, :2]
    t_iod = float(np.linalg.norm(t_lm[EYE_R].mean(0) - t_lm[EYE_L].mean(0)))

    A_sim = umeyama(f_lm, t_lm)
    sc = float(np.sqrt(abs(np.linalg.det(A_sim[:2, :2]))))

    anch = boundary_anchors(ALIGN_N_BOUNDARY)
    to_tgt = lambda p: (A_sim @ np.hstack([p, np.ones((len(p), 1))]).T).T[:, :2]
    anch_in_tgt = to_tgt(anch)
    ctrl_src = np.vstack([to_tgt(f_lm), anch_in_tgt])
    ctrl_dst = np.vstack([t_lm, anch_in_tgt])

    c_norm, s_norm = t_lm.mean(0), t_iod
    csn = (ctrl_src - c_norm) / s_norm
    params = tps_fit(csn, (ctrl_dst - c_norm) / s_norm, ALIGN_TPS_LAMBDA)

    yy, xx = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64),
                         indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel()], 1)
    mp = to_tgt(pts)
    mp = (tps_apply_np(params, csn, (mp - c_norm) / s_norm) * s_norm + c_norm).reshape(H, W, 2)

    src_t = torch.from_numpy(np.asarray(target_rgb_u8, np.float32)).permute(2, 0, 1)[None].to(device)
    Ht, Wt = src_t.shape[-2:]
    if sc > ALIGN_ANTIALIAS_THRESH:
        src_t = F.interpolate(src_t, scale_factor=1.0 / sc, mode="area")
        Hs, Ws = src_t.shape[-2:]

        mp[..., 0] = (mp[..., 0] + 0.5) * (Ws / Wt) - 0.5
        mp[..., 1] = (mp[..., 1] + 0.5) * (Hs / Ht) - 0.5
    Hs, Ws = src_t.shape[-2:]

    gx = torch.from_numpy(mp[..., 0]).float().to(device)
    gy = torch.from_numpy(mp[..., 1]).float().to(device)
    grid = torch.stack([(2.0 * gx + 1.0) / Ws - 1.0, (2.0 * gy + 1.0) / Hs - 1.0], -1)[None]
    with torch.no_grad():
        warped = F.grid_sample(src_t, grid, mode="bilinear", align_corners=False,
                               padding_mode="border")
    return np.clip(warped[0].permute(1, 2, 0).cpu().numpy(), 0, 255).astype(np.float32)

def tps_track_grids(lm68, device):
    lm = np.asarray(lm68, np.float64)
    T = int(lm.shape[0])
    if not np.isfinite(lm[ANCHOR]).all():
        raise RuntimeError(f"anchor frame {ANCHOR} has no 68-point landmarks; TPS tracking is defined relative to the anchor, so without it the whole clip would be misaligned")
    b = boundary_anchors(TRACK_N_BOUNDARY)
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64),
                         indexing="ij")
    pts_t = torch.as_tensor(np.stack([xx.ravel(), yy.ravel()], 1), dtype=torch.float64, device=device)

    grids = [None] * T
    todo = [t for t in range(T) if np.isfinite(lm[t]).all()]
    for t in todo:

        cs = np.vstack([lm[t], b])
        cd = np.vstack([lm[ANCHOR], b])
        c = cs.mean(0)
        sn = max(float(np.abs(cs - c).max()), 1e-6)
        csn = (cs - c) / sn
        params = tps_fit(csn, (cd - c) / sn, TRACK_TPS_LAMBDA)
        cq = torch.as_tensor(c, dtype=torch.float64, device=device)
        mp = tps_apply_torch(params, csn, (pts_t - cq) / sn, device) * sn + cq
        gx = (2 * mp[:, 0].reshape(H, W) + 1) / W - 1
        gy = (2 * mp[:, 1].reshape(H, W) + 1) / H - 1
        grids[t] = torch.stack([gx, gy], -1)[None].to(torch.float32)
    return grids

def warp(x, grid):
    return x if grid is None else F.grid_sample(x, grid, align_corners=False,
                                                padding_mode="zeros", mode="bilinear")

def build_face_delta(clean_anchor_u8, target_aligned, mask):
    from skimage.color import lab2rgb, rgb2lab

    src = np.asarray(clean_anchor_u8, np.float32)
    lab0 = rgb2lab(src / 255.0)
    lab1 = rgb2lab(np.clip(np.asarray(target_aligned, np.float32) / 255.0, 0, 1))
    d = (lab1 - lab0).astype(np.float32)

    d[..., 0] *= LAB_W_L
    d[..., 1] *= LAB_W_AB
    d[..., 2] *= LAB_W_AB

    d = d - cv2.GaussianBlur(d, (0, 0), LAB_SIGMA_HI)

    delta = lab2rgb(lab0 + LAB_GAIN * d) * 255.0 - src
    delta = EPS_PIXEL * np.sign(delta)
    delta = np.clip(delta, -EPS_PIXEL, EPS_PIXEL) * np.asarray(mask, np.float32)[..., None]
    return np.round(delta).astype(np.int16)

def select_training_frames(clip_u8):
    T = clip_u8.shape[0]
    gray = [cv2.resize(cv2.cvtColor(clip_u8[t], cv2.COLOR_RGB2GRAY),
                       (clip_u8.shape[2] // 2, clip_u8.shape[1] // 2)) for t in range(T)]
    mags = np.zeros(T)
    for t in range(1, T):
        flow = cv2.calcOpticalFlowFarneback(gray[t - 1], gray[t], flow=None, pyr_scale=0.5,
                                            levels=3, winsize=15, iterations=3, poly_n=5,
                                            poly_sigma=1.2, flags=0)
        mags[t] = np.linalg.norm(flow, axis=2).mean()
    mags[0] = mags[1]
    mags[T - 1] = mags[T - 2]

    cum = np.cumsum(mags.astype(np.float64))
    cum /= (cum[-1] + 1e-12)
    edges = [0] + [int(np.searchsorted(cum, q / MF_NUM)) for q in range(1, MF_NUM)] + [T]
    edges = sorted(set(e for e in edges if 0 <= e <= T))
    return [s + int(np.argmax(mags[s:e])) for s, e in zip(edges[:-1], edges[1:])]

def protect(videos: dict) -> dict:
    pipe = _get_pipe()
    _get_face_app()
    _get_mask_app()
    device = next(pipe.transformer.parameters()).device

    target_path = Path(TARGET_IMAGE)
    if not target_path.exists():
        raise FileNotFoundError(f"target-identity image not found: {target_path} (override with SYNTH_TARGET)")
    target_rgb = np.array(Image.open(target_path).convert("RGB"))

    out = {}
    for name, v in videos.items():
        T = v.shape[0]

        M_soft, _ = build_face_union_mask(v)
        M_hard = (M_soft > 0.5).astype(np.float32)

        lm68 = track_landmarks(v)
        aligned = align_target_to_anchor(target_rgb, v[ANCHOR], str(device))
        d_face = build_face_delta(v[ANCHOR], aligned, M_soft)

        grids = tps_track_grids(lm68, device)
        Mh_t = torch.tensor(M_hard, device=device).view(1, 1, H, W)
        dF = torch.tensor(np.asarray(d_face, np.float32)).permute(2, 0, 1)[None].to(device)
        m_ts = []
        fb = np.zeros((T, H, W, 3), np.float32)
        um = np.zeros((T, H, W), np.float32)
        with torch.no_grad():
            for t in range(T):

                m_t = (warp(Mh_t, grids[t]) > 0.5).float() if grids[t] is not None else Mh_t
                m_ts.append(m_t)
                fb[t] = (warp(dF, grids[t]) * m_t)[0].permute(1, 2, 0).cpu().numpy()
                um[t] = (1.0 - m_t)[0, 0].cpu().numpy()
        idxs = select_training_frames(v)

        random.seed(SEED); np.random.seed(SEED)
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

        d_bg = attack_background_delta(
            pipe, [Image.fromarray(v[i]) for i in idxs],
            frozen_base=torch.from_numpy(
                np.ascontiguousarray(fb[idxs].transpose(0, 3, 1, 2) / 127.5)),
            update_mask=torch.from_numpy(
                np.ascontiguousarray(um[idxs].reshape(len(idxs), 1, H, W))))

        dB = torch.tensor(np.asarray(d_bg, np.float32)).permute(2, 0, 1)[None].to(device)
        protected = np.empty_like(v)
        with torch.no_grad():
            for t in range(T):
                de = torch.clamp(dB * (1.0 - m_ts[t]) + warp(dF, grids[t]) * m_ts[t],
                                 -EPS_PIXEL, EPS_PIXEL)
                p = torch.clamp(torch.tensor(v[t], device=device).permute(2, 0, 1)[None].float() + de,
                                0, 255)
                protected[t] = p[0].permute(1, 2, 0).round().to(torch.uint8).cpu().numpy()

        linf = int(np.abs(protected.astype(np.int16) - v.astype(np.int16)).max())
        assert linf <= EPS_PIXEL, f"clip '{name}': L_inf={linf} exceeds the budget {EPS_PIXEL}"
        out[name] = protected

        del dB, dF, Mh_t, m_ts
        torch.cuda.empty_cache()
    return out
