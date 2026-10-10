from __future__ import annotations
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import math
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from diffusers.models.transformers.transformer_wan import WanAttnProcessor

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.use_deterministic_algorithms(True, warn_only=True)


MODEL_DIR_DEFAULT = os.environ.get("WAN_MODEL_PATH", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")

_MF_NUM = 5
_EOT_PROMPTS_LIST = [
    "A young man speaks expressively to the camera against a green backdrop.",
    "a young man speaks expressively to the camera against a green backdrop",
    "a young man speaking to the camera, green background",
    "a man talking expressively to the camera on a green backdrop",
    "a person speaking to the camera against a green screen",
    "a young man talking to the camera",
]

_CAPTURED_W22: dict[int, dict[str, torch.Tensor]] = {}
_CAPTURED_CROSS_W22: dict[int, dict[str, torch.Tensor]] = {}
_S_PER_FRAME: int = 0  


def load_wan22_pipe(model_dir: str = MODEL_DIR_DEFAULT,
                    device: str = 'cuda',
                    dtype: torch.dtype = torch.bfloat16) -> WanImageToVideoPipeline:

    vae = AutoencoderKLWan.from_pretrained(model_dir, subfolder='vae', torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(model_dir, vae=vae, torch_dtype=dtype).to(device)

    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.transformer.enable_gradient_checkpointing()
    for module in (pipe.vae, pipe.transformer, pipe.text_encoder):
        for p in module.parameters():
            p.requires_grad_(False)
    return pipe


def encode_image_to_first_frame_latent(pipe: WanImageToVideoPipeline,
                                         img_neg1pos1: torch.Tensor,
                                         dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:


    video_condition = img_neg1pos1.unsqueeze(2).to(dtype=pipe.vae.dtype)


    encoded = pipe.vae.encode(video_condition)
    latent_condition = encoded.latent_dist.mode().to(dtype)


    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean, dtype=dtype, device=latent_condition.device)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
    )
    latents_std_inv = (
        1.0 / torch.tensor(pipe.vae.config.latents_std, dtype=dtype, device=latent_condition.device)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
    )
    latent_condition = (latent_condition - latents_mean) * latents_std_inv
    return latent_condition


def decode_latents_to_video(pipe: WanImageToVideoPipeline,
                            latents: torch.Tensor) -> torch.Tensor:

    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean, dtype=latents.dtype, device=latents.device)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
    )
    latents_std = (
        torch.tensor(pipe.vae.config.latents_std, dtype=latents.dtype, device=latents.device)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
    )
    latents_unnorm = latents * latents_std + latents_mean
    return pipe.vae.decode(latents_unnorm.to(pipe.vae.dtype)).sample


def compute_temp_ts(first_frame_mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    sliced = first_frame_mask[0, 0, :, ::2, ::2]
    flat = (sliced * t).flatten()
    return flat.unsqueeze(0)


class CaptureWanAttnProcessor(WanAttnProcessor):
    def __init__(self, block_id: int):
        super().__init__()
        self.block_id = block_id

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None):

        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        query_n = attn.norm_q(query) if attn.norm_q is not None else query
        key_n = attn.norm_k(key) if attn.norm_k is not None else key

        query_h = query_n.unflatten(2, (attn.heads, -1))
        key_h = key_n.unflatten(2, (attn.heads, -1))
        value_h = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

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


        _CAPTURED_W22[self.block_id] = {
            'q': query_h.flatten(2, 3),
            'k': key_h.flatten(2, 3),
            'v': value,
        }


        from diffusers.models.attention_dispatch import dispatch_attention_fn
        out = dispatch_attention_fn(
            query_h, key_h, value_h,
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            backend=self._attention_backend,
        )
        out = out.flatten(2, 3).type_as(query_h)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


def install_capture_patches_w22(transformer, block_ids):
    for bid in block_ids:
        transformer.blocks[bid].attn1.set_processor(CaptureWanAttnProcessor(bid))


def install_cross_capture_w22(transformer, block_ids):
    for bid in block_ids:
        transformer.blocks[bid].attn2.set_processor(CaptureCrossAttnProcessor(bid))


class CaptureCrossAttnProcessor(WanAttnProcessor):
    def __init__(self, block_id: int):
        super().__init__()
        self.block_id = block_id

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None):
        out = super().__call__(attn, hidden_states, encoder_hidden_states,
                               attention_mask, rotary_emb)
        _CAPTURED_CROSS_W22[self.block_id] = {'out': out}
        return out


def partial_dit_forward_w22(transformer, hidden_states, timestep, encoder_hidden_states,
                             target_block_max):
    rotary_emb = transformer.rope(hidden_states)
    hidden_states = transformer.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
    ts_seq_len = timestep.shape[1]
    _, timestep_proj, encoder_hidden_states_emb, _ = transformer.condition_embedder(
        timestep.flatten(), encoder_hidden_states, None, timestep_seq_len=ts_seq_len)
    timestep_proj = timestep_proj.unflatten(2, (6, -1))
    target = min(target_block_max, len(transformer.blocks) - 1)
    for i in range(target + 1):
        hidden_states = transformer._gradient_checkpointing_func(
            transformer.blocks[i], hidden_states, encoder_hidden_states_emb, timestep_proj, rotary_emb)
    return hidden_states


class RandomDownsampleUpsample(torch.nn.Module):
    def __init__(self, scale_range=(0.3, 0.7), modes=('bilinear', 'bicubic', 'area')):
        super().__init__()
        self.scale_range = scale_range
        self.modes = modes
    def forward(self, x):
        H, W = x.shape[-2:]
        s = random.uniform(*self.scale_range)
        h_low = max(8, int(H * s)); w_low = max(8, int(W * s))
        m_down = random.choice(self.modes)
        m_up = random.choice(self.modes)
        x_low = F.interpolate(x, size=(h_low, w_low), mode=m_down,
                              align_corners=False if m_down != 'area' else None)
        x_back = F.interpolate(x_low, size=(H, W), mode=m_up,
                               align_corners=False if m_up != 'area' else None)
        return x_back.clamp(0.0, 1.0)


def capacity_ratio_F0(v_full):
    V_F0 = v_full[0, :_S_PER_FRAME, :]
    v_mean = V_F0.mean(dim=0, keepdim=True)
    V_centered = V_F0 - v_mean
    capacity = (V_centered ** 2).sum()
    total = (V_F0 ** 2).sum() + 1e-8
    return capacity / total


def k_attract_loss(q_full, k_full, num_heads, head_dim):
    q = q_full.detach()
    k = k_full
    B, N, _ = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    q_h = q.view(B, N, num_heads, head_dim)
    k_h = k.view(B, N, num_heads, head_dim)
    q_F1plus = q_h[:, _S_PER_FRAME:]
    k_F0 = k_h[:, :_S_PER_FRAME]
    q_pool = q_F1plus.mean(dim=1)
    k_pool = k_F0.mean(dim=1)
    inner = (q_pool * k_pool).sum(dim=-1) * scale
    return inner.mean()


def vae_l2_dist(za, zc):
    return ((za.float() - zc.float()) ** 2).mean()


def vae_latent_struct(za, zc):
    from kornia.losses import ssim_loss as _struct_loss
    a = za.float(); b = zc.float()
    B, C, F, H, W = a.shape
    a = a.reshape(B * C * F, 1, H, W)
    b = b.reshape(B * C * F, 1, H, W)
    lo = b.detach().amin(dim=(2, 3), keepdim=True)
    hi = b.detach().amax(dim=(2, 3), keepdim=True)
    scale = (hi - lo).clamp_min(1e-3)
    a = (a - lo) / scale
    b = (b - lo) / scale
    return _struct_loss(a, b, window_size=7, max_val=1.0)


def _gaussian_window2d(ksize, sigma, device, dtype):
    ax = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    g = torch.exp(-(ax * ax) / (2.0 * sigma * sigma))
    g = g / g.sum()
    k = torch.outer(g, g)
    return (k / k.sum()).view(1, 1, ksize, ksize)


def _luma255(I_neg1pos1):
    I01 = (I_neg1pos1.float() + 1.0) * 0.5
    y = 0.299 * I01[:, 0:1] + 0.587 * I01[:, 1:2] + 0.114 * I01[:, 2:3]
    return 255.0 * y


def mscn_map(gray255, ksize=7, sigma=7.0 / 6.0):
    k = _gaussian_window2d(ksize, sigma, gray255.device, gray255.dtype)
    pad = ksize // 2
    mu = F.conv2d(gray255, k, padding=pad)
    mu_sq = F.conv2d(gray255 * gray255, k, padding=pad)
    var = torch.clamp(mu_sq - mu * mu, min=0.0)
    sig = torch.sqrt(var + 1e-6)
    return (gray255 - mu) / (sig + 1.0)


def mscn_unnaturalness(I_neg1pos1, ksize=7):
    gray = _luma255(I_neg1pos1)
    m = mscn_map(gray, ksize=ksize)
    m = m[..., 1:-1, 1:-1]
    denom = (m * m).mean() + 1e-6
    prods = [
        m[..., :, :-1] * m[..., :, 1:],
        m[..., :-1, :] * m[..., 1:, :],
        m[..., :-1, :-1] * m[..., 1:, 1:],
        m[..., :-1, 1:] * m[..., 1:, :-1],
    ]
    corr = torch.stack([p.mean() for p in prods]).mean() / denom
    return corr


def text_div_collapse_loss(text_attn, K=400):
    t = text_attn[0].float()
    N = t.shape[0]
    K = min(K, N)
    idx = torch.randperm(N, device=t.device)[:K]
    sub = t[idx]
    sub_n = sub / (sub.norm(dim=-1, keepdim=True) + 1e-8)
    cos_mat = sub_n @ sub_n.T
    mean_cos = (cos_mat.sum() - K) / (K * (K - 1))
    return -mean_cos


def preprocess_img(pil, h, w):
    pil = pil.convert("RGB").resize((w, h))
    arr = np.asarray(pil).astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    return ten


def encode_prompt_w22(pipe: WanImageToVideoPipeline, prompt: str, device='cuda', dtype=torch.bfloat16):

    pos_emb, _ = pipe.encode_prompt(
        prompt=prompt, do_classifier_free_guidance=False, num_videos_per_prompt=1,
        device=device, dtype=dtype,
    )
    return pos_emb


def attack_universal_delta(pipe, src_pils, h, w, num_frames, num_steps, epsilon, step_size,
                           timestep_low, timestep_high, seed=0, num_inference_steps_attack=25,
                           eot_layers=None, eot_prompts=None, momentum_beta=0.9, eot_samples=1,
                           k_lambda=0.5, vae_l2=0.0, vae_struct=0.0, brisque=0.0, cross_lambda=0.0):
    global _S_PER_FRAME
    _S_PER_FRAME = (h // 16 // 2) * (w // 16 // 2)
    num_layers = len(pipe.transformer.blocks)
    num_heads = pipe.transformer.config.num_attention_heads
    head_dim = pipe.transformer.config.attention_head_dim
    N = len(src_pils)

    import kornia.augmentation as KA
    eot_augs = [
        ("identity",      None),
        ("jpeg",          KA.RandomJPEG(jpeg_quality=(50.0, 70.0), p=1.0).to('cuda')),
        ("blur",          KA.RandomGaussianBlur(kernel_size=(7, 7), sigma=(1.0, 3.0), p=1.0).to('cuda')),
        ("noise",         KA.RandomGaussianNoise(mean=0.0, std=0.05, p=1.0).to('cuda')),
        ("downsample_up", RandomDownsampleUpsample(scale_range=(0.3, 0.7)).to('cuda')),
    ]
    active_blocks = [min(b, num_layers - 1) for b in eot_layers]
    install_capture_patches_w22(pipe.transformer, active_blocks)
    install_cross_capture_w22(pipe.transformer, active_blocks)

    with torch.no_grad():
        prompt_emb_list = [{'pos': encode_prompt_w22(pipe, p)} for p in eot_prompts]

    pipe.scheduler.set_timesteps(num_inference_steps_attack, device=pipe.device)
    lH = h // 16; lW = w // 16
    lF = (num_frames - 1) // 4 + 1
    noise_gen = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn((1, 48, lF, lH, lW), generator=noise_gen, device='cpu',
                        dtype=torch.float32).to(device=pipe.device, dtype=pipe.transformer.dtype)
    first_frame_mask = torch.ones(1, 1, lF, lH, lW, dtype=pipe.transformer.dtype, device=pipe.device)
    first_frame_mask[:, :, 0] = 0

    I_before_list = [preprocess_img(p, h, w).to(device=pipe.device, dtype=pipe.transformer.dtype)
                     for p in src_pils]
    with torch.no_grad():
        z_c_list = [encode_image_to_first_frame_latent(pipe, ib, dtype=pipe.transformer.dtype)
                    for ib in I_before_list]

    delta = torch.empty_like(I_before_list[0]).uniform_(-epsilon / 4, epsilon / 4)
    delta = delta.detach().requires_grad_(True)
    momentum = torch.zeros_like(delta)

    for step in range(num_steps):
        if delta.grad is not None:
            delta.grad.zero_()
        for _k in range(eot_samples):
            j = random.randrange(N)
            I_adv_j = torch.clamp(I_before_list[j] + delta, -1.0, 1.0)
            sampled_aug = random.choice(eot_augs)
            if sampled_aug[1] is None:
                I_adv_aug = I_adv_j
            else:
                x01 = ((I_adv_j + 1.0) * 0.5).clamp(0.0, 1.0).to(torch.float32)
                x01 = sampled_aug[1](x01)
                I_adv_aug = (x01 * 2.0 - 1.0).clamp(-1.0, 1.0).to(I_adv_j.dtype)
            image_latent_adv = encode_image_to_first_frame_latent(pipe, I_adv_aug, dtype=pipe.transformer.dtype)

            loss = torch.zeros(1, device=pipe.device, dtype=pipe.transformer.dtype).squeeze()
            sampled_block = random.choice(active_blocks)
            sampled_prompt_emb = random.choice(prompt_emb_list)
            t_idx = random.randint(timestep_low, timestep_high)
            t_scalar = pipe.scheduler.timesteps[t_idx].to(pipe.device)
            pipe.scheduler.set_timesteps(num_inference_steps_attack, device=pipe.device)
            with torch.no_grad():
                cur = noise.clone()
                for i in range(t_idx):
                    tt_i = pipe.scheduler.timesteps[i].to(pipe.device)
                    latent_input = (1 - first_frame_mask) * image_latent_adv + first_frame_mask * cur
                    temp_ts = compute_temp_ts(first_frame_mask, tt_i)
                    v_pred_i = pipe.transformer(hidden_states=latent_input, timestep=temp_ts,
                                                encoder_hidden_states=sampled_prompt_emb['pos'],
                                                return_dict=False)[0]
                    cur = pipe.scheduler.step(v_pred_i, pipe.scheduler.timesteps[i], cur).prev_sample
            latent_input = (1 - first_frame_mask) * image_latent_adv + first_frame_mask * cur
            temp_ts = compute_temp_ts(first_frame_mask, t_scalar)

            _CAPTURED_W22.clear()
            partial_dit_forward_w22(pipe.transformer, latent_input, temp_ts,
                                      sampled_prompt_emb['pos'], target_block_max=sampled_block)
            cell = _CAPTURED_W22[sampled_block]
            loss = loss + capacity_ratio_F0(cell['v'].float())
            loss = loss - k_lambda * k_attract_loss(
                cell['q'].float(), cell['k'].float(), num_heads=num_heads, head_dim=head_dim)
            loss = loss + cross_lambda * text_div_collapse_loss(_CAPTURED_CROSS_W22[sampled_block]['out'])
            loss = loss - vae_l2 * vae_l2_dist(image_latent_adv, z_c_list[j])
            loss = loss - vae_struct * vae_latent_struct(image_latent_adv, z_c_list[j])
            recon = decode_latents_to_video(pipe, image_latent_adv)[:, :, 0]
            loss = loss - brisque * mscn_unnaturalness(recon)

            (loss / eot_samples).backward()

        grad_sign = delta.grad.data.sign()
        momentum = momentum_beta * momentum + grad_sign
        delta.data = delta.data - step_size * momentum.sign()
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.requires_grad_(True)

    return delta.detach()


CFG = dict(
    h=480, w=640,
    num_frames=17,
    epsilon=16 / 255 * 2,
    num_steps=90,
    eot_layers=[0, 3, 6],
    t_low=5, t_high=5,
    num_inference_steps_attack=25,
    k_lambda=0.5,
    momentum_beta=0.9,
    eot_samples=3,
    seed=0,
    vae_l2=1.0,
    vae_struct=1.0,
    brisque=2.0,
    cross_lambda=1.0,
)

EPS_PIXEL = 16
_PIPE = None


def _get_pipe():
    global _PIPE
    if _PIPE is None:
        print("[my_method] loading Wan2.2-TI2V-5B (white-box source == eval model) ...")
        _PIPE = load_wan22_pipe()
    return _PIPE


def compute_optical_flow_magnitudes(
    frames_rgb: "np.ndarray",
    downsample_ratio: float = 0.5,
) -> np.ndarray:

    T = frames_rgb.shape[0]
    frames_gray = []
    for t in range(T):
        gray = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2GRAY)
        if downsample_ratio != 1.0:
            h_new = int(gray.shape[0] * downsample_ratio)
            w_new = int(gray.shape[1] * downsample_ratio)
            gray = cv2.resize(gray, (w_new, h_new))
        frames_gray.append(gray)

    if T < 2:
        raise ValueError(f"Video has only {T} frame(s); need at least 2.")

    flow_magnitudes = np.zeros(T)
    for t in range(1, T):
        flow = cv2.calcOpticalFlowFarneback(
            frames_gray[t - 1], frames_gray[t], flow=None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        flow_magnitudes[t] = np.linalg.norm(flow, axis=2).mean()

    flow_magnitudes[0] = flow_magnitudes[1]
    flow_magnitudes[T - 1] = flow_magnitudes[T - 2]
    return flow_magnitudes


def make_windows_energy(mags, K, max_win):
    T = len(mags)
    K = max(1, min(K, max_win, T))
    cum = np.cumsum(np.asarray(mags, dtype=np.float64))
    cum /= (cum[-1] + 1e-12)
    edges = [0]
    for q in range(1, K):
        edges.append(int(np.searchsorted(cum, q / K)))
    edges.append(T)
    edges = sorted(set(e for e in edges if 0 <= e <= T))
    return [(edges[k], edges[k + 1]) for k in range(len(edges) - 1)]


def _frame_set(v):
    mags = compute_optical_flow_magnitudes(v)
    windows = make_windows_energy(mags, _MF_NUM, _MF_NUM)
    return [s + int(np.argmax(mags[s:e])) for (s, e) in windows]


def protect(videos: dict) -> dict:
    pipe = _get_pipe()
    out = {}
    for name, v in videos.items():
        idxs = _frame_set(v)
        print(f"[my_method] clip '{name}': training shared δ on frames {idxs} "
              f"({CFG['num_steps']} steps, eot_samples={CFG['eot_samples']})", flush=True)

        fseed = CFG["seed"] * 100003
        random.seed(fseed); np.random.seed(fseed % (2 ** 32))
        torch.manual_seed(fseed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fseed)

        src_pils = [Image.fromarray(v[i]) for i in idxs]
        step_size = CFG["epsilon"] / 20
        t0 = time.time()
        delta = attack_universal_delta(
            pipe, src_pils, CFG["h"], CFG["w"], CFG["num_frames"],
            CFG["num_steps"], CFG["epsilon"], step_size, CFG["t_low"], CFG["t_high"],
            seed=fseed, num_inference_steps_attack=CFG["num_inference_steps_attack"],
            eot_layers=CFG["eot_layers"], eot_prompts=_EOT_PROMPTS_LIST,
            momentum_beta=CFG["momentum_beta"], eot_samples=CFG["eot_samples"],
            k_lambda=CFG["k_lambda"], vae_l2=CFG["vae_l2"], vae_struct=CFG["vae_struct"],
            brisque=CFG["brisque"], cross_lambda=CFG["cross_lambda"],
        )

        d_px = np.round(delta.float().cpu().numpy()[0].transpose(1, 2, 0) * 127.5).astype(np.int16)
        d_px = np.clip(d_px, -EPS_PIXEL, EPS_PIXEL)
        protected = np.clip(v.astype(np.int16) + d_px[None], 0, 255).astype(np.uint8)
        eps = int(np.abs(protected.astype(np.int16) - v.astype(np.int16)).max())
        print(f"[my_method] clip '{name}' done in {time.time()-t0:.1f}s: "
              f"δ_Linf={int(np.abs(d_px).max())} clip_Linf={eps} (budget {EPS_PIXEL})", flush=True)
        out[name] = protected
    return out
