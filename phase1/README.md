# Phase 1

Code for our Phase 1 submission. Setup and the commands to run it are in the
[top-level README](../README.md); this page covers the Phase 1 setting and the details specific to
this phase.

- **Method file:** `protect/methods/my_method.py` (the whole protection, self-contained)
- **Entry point:** `python protect/run.py --method my_method`
- **Output:** `data/protected/my_method/submission/M003/front_neutral_level_1.mp4`

## Table of Contents

- [Challenge Setting](#challenge-setting)
- [Method](#method)
- [Hardware](#hardware)
- [Repository Structure](#repository-structure)
- [Tested Environment](#tested-environment)

## Challenge Setting

- **Data:** one face clip of MEAD identity `M003` (`front_neutral_level_1`, 113 frames, 640×480).
- **Budget:** `L∞ ≤ 16/255` per pixel.
- **Threat model (reference-based):** the attacker samples one frame from the protected clip and
  feeds it, together with a text prompt, to an image-to-video (I2V) model.
- **Known model:** [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers), with
  white-box access.
- **Tracks:** *Effectiveness* (the protected clip as-is) and *Robustness* (after three hidden
  temporal attacks). The prompts, the reference-frame sampling strategy, the scoring models and the
  temporal attacks are hidden.
- **Score:** invisibility plus the harmonic mean of the four tracks; see
  [`eval_competition/README.md`](eval_competition/README.md) for the metric definitions.

## Method

**Reference Capacity Collapse.** Two complementary losses on the self-attention of Wan2.2:
**V-capacity collapse** makes the reference frame's Values spatially homogeneous, which reduces
the information the reference can provide; **K-attract** makes the reference Keys better match the
(stop-gradient) Queries of the following frames, so later frames keep depending on the degraded
reference. A few
auxiliary objectives further strengthen the perturbation.

**Motion-Guided EOT Optimization.** One video-level perturbation is shared by all frames and
learned from five motion-selected representative frames. Expectation-over-Transformation
randomizes the reference frame, input transformation, prompt and Transformer block.

All hyper-parameters are hard-coded at the top of `protect/methods/my_method.py`.

## Hardware

- **GPU VRAM: ≥ 40 GB recommended** (observed peak ≈ 37 GB). Produced on a single NVIDIA H100;
  a 24 GB card is likely to run out of memory.
- **Runtime:** ≈ 6 minutes for the clip (≈ 335 s on H100).

## Repository Structure

```
phase1/
├── README.md                     # this file
├── requirements.txt              # Python dependencies
├── protect/
│   ├── run.py                    # entry point: --method my_method
│   └── methods/
│       ├── my_method.py          # >>> OUR METHOD <<<
│       ├── random_noise.py       # kit baseline (uniform L-inf noise)
│       └── no_protection.py      # kit baseline (identity)
├── data_preparation/             # download check + preprocessing of the M003 clip
├── eval/                         # kit's local reference evaluation
├── eval_competition/             # leaderboard metric definitions + reproduction
├── utils/                        # io_video / i2v / metrics helpers
├── check_submission.py           # validates submission.zip
└── data/                         # created when running (original/, preprocessed/, protected/, generated/)
```

## Tested Environment

Install with `pip install -r requirements.txt` followed by `pip install "diffusers==0.37.1"`; newer
diffusers releases change the Wan2.2 pipeline/attention internals. The other packages were tested
at the versions below.

| package | version |
| --- | --- |
| diffusers | **0.37.1** (pin this) |
| torch | 2.10.0+cu128 |
| kornia | 0.8.2 |
| opencv-python | 4.11 and 4.13 |
| numpy | 1.26 (as installed by `requirements.txt`, `numpy<2`) and 2.2.6 |
| decord | 0.6.0 |

> **Note.** Bit-exact reproduction is not guaranteed. The random downsample-upsample EOT
> augmentation uses `area` and `bicubic` interpolation, whose CUDA backward passes have no
> deterministic implementation, so a rerun can end at a slightly different perturbation.
