# KVeil (ECCV 2026 LifeGenIP Challenge, 1st Place)

This repository contains the official implementation for "KVeil: First-Place Solution to ECCV 2026 LifeGenIP Challenge on Unlearnable Videos", our first-place solution to both phases of the [ECCV 2026 LifeGenIP Challenge on Unlearnable Videos against Diffusion-based Customization](https://lifegenip.cc/competition).

The challenge asks for an imperceptible perturbation (`L∞ ≤ 16/255`) on face videos that stops them from being used for identity customization by video diffusion models.

- **Phase 1** (reference-based customization): **Reference Capacity Collapse** attacks the self-attention of the I2V model with two complementary losses. **V-capacity collapse** reduces the information that the reference frame can provide, and **K-attract** keeps the following frames dependent on that degraded reference. **Motion-Guided EOT** adds robustness to unknown inference conditions.
- **Phase 2** (reference-based + tuning-based customization, unseen models): a two-layer framework. **Prior-Aligned Face Protection** places a structure-aligned perturbation built from a synthetic target identity on the face. **White-Box Background Protection** applies the Phase 1 attack (V-capacity collapse + K-attract) to the remaining region.

The code is built on the official [challenge starting kit](https://github.com/tmllab/ECCV26_LifeGenIP_starting_kit). Each phase lives in its own directory, and the whole protection method is a single self-contained file:

| Phase | Method file | Phase notes (setting, hardware, environment) |
|---|---|---|
| Phase 1 | `phase1/protect/methods/my_method.py` | [`phase1/README.md`](phase1/README.md) |
| Phase 2 | `phase2/protect/methods/my_method.py` (+ `assets/original.png`) | [`phase2/README.md`](phase2/README.md) |

## Setup Instructions
This section explains how to prepare the environment. The two phases use separate environments, because Phase 2 pins its own dependency versions.

### 1. Environment Setup
#### Phase 1
Create and activate a Conda environment with Python 3.10:
```
conda create --name lifegenip python=3.10 -y
conda activate lifegenip
conda install -c conda-forge ffmpeg -y
```
#### Phase 2
```
conda create --name lifegenip-protect python=3.10 -y
conda activate lifegenip-protect
conda install -c conda-forge ffmpeg=5.0.1 -y
```

### 2. Installation of Dependencies
#### Phase 1
```
cd phase1
pip install -r requirements.txt
pip install "diffusers==0.37.1"   # the version the submission was produced with
```
#### Phase 2
```
cd phase2
pip install -r requirements-protect.txt
```
Let ONNX Runtime use the CUDA libraries shipped with the pip `torch` wheel, so the InsightFace face detector runs on the GPU:
```
export LD_LIBRARY_PATH="$(python -c "import os,glob,torch;\
p=os.path.join(os.path.dirname(torch.__file__),'..','nvidia');\
print(':'.join(sorted(glob.glob(os.path.join(p,'*','lib')))))")":$LD_LIBRARY_PATH
```
The Phase 2 local fine-tuning evaluation (optional) also needs [ai-toolkit](https://github.com/ostris/ai-toolkit) at the commit pinned by the starting kit. Run this inside `phase2/` (the evaluator looks for it at `phase2/third_party/ai-toolkit`):
```
git clone https://github.com/ostris/ai-toolkit.git third_party/ai-toolkit
git -C third_party/ai-toolkit checkout fed9357234f12da9e6e00cc6b3e20b34a1369843
```

## Quickstart Guidelines
### 1. Dataset & Pre-trained Models Preparation
### Datasets
Both phases use identity `M003` from the [MEAD dataset](https://wywu.github.io/projects/MEAD/MEAD.html). Download only `M003/video.tar` (see the [MEAD repo](https://github.com/uniBruce/Mead)), place it at `data/original/video.tar` inside the phase directory, and preprocess:
```
python data_preparation/check.py        # optional: verify video.tar
python data_preparation/preprocess.py
```
This writes the clean clips (113 frames, 640×480) to `data/preprocessed/M003/`. Phase 1 uses 1 clip and Phase 2 uses 10 clips.

### Models
Both phases attack [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) as the white-box model. Download it and export its path:
```
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --local-dir ./models/Wan2.2-TI2V-5B-Diffusers
export WAN_MODEL_PATH=$PWD/models/Wan2.2-TI2V-5B-Diffusers
```
If `WAN_MODEL_PATH` is unset, the code falls back to the Hugging Face hub id and downloads on first use.

Phase 2 also needs the InsightFace packs `antelopev2` and `buffalo_l`. They are downloaded automatically into `~/.insightface/` on first use (set `ANTELOPEV2_ROOT` to use another location). If `antelopev2` unpacks into a nested `antelopev2/antelopev2/` folder, move its `.onnx` files up one level (see [`phase2/README.md`](phase2/README.md#face-models)).

#### Target Identity Image (Phase 2)
The face layer pushes the face towards a synthetic target identity, `phase2/protect/methods/assets/original.png`. `t_1.png` to `t_5.png` in the same folder are five images regenerated with a fixed prompt, provided as a reproducibility check (they reach a similar ISM). See [Target Identity Image](phase2/README.md#target-identity-image--generation-and-reproducibility) for the prompt and results.

### 2. Protection
All hyper-parameters are hard-coded at the top of each `my_method.py`. There are no command-line flags or config files.

#### Phase 1
Protect the single Phase 1 clip (≈ 6 min on one H100; ≥ 40 GB VRAM recommended):
```
cd phase1
python protect/run.py --method my_method
```

#### Phase 2
Protect all 10 Phase 2 clips (≈ 9 min per clip on one H100; ≥ 32 GB VRAM recommended):
```
cd phase2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python protect/run.py --method my_method
```
The clips are independent (the RNGs are re-seeded per clip), so the run can be split across GPUs.

Both phases write two copies of each protected clip:
- `data/protected/my_method/submission/M003/<clip>.mp4`: losslessly encoded (no compression loss on the protected frames); use this one for evaluation and submission.
- `data/protected/my_method/preview/M003/<clip>.mp4`: lossy h264, for viewing.

### 3. Evaluation (Optional)
#### Generate Videos
Use the starting kit's local evaluation, run in the kit's own environment (see the [starting kit](https://github.com/tmllab/ECCV26_LifeGenIP_starting_kit) for setup). It generates videos from the protected clips: I2V in both phases, plus LoRA fine-tuning in Phase 2. It also reports the kit's reference metrics.
```
python eval/run.py --method my_method
```

#### Score with the Leaderboard Metrics
After the challenge, the organizers published the definitions of the leaderboard metrics (`eval_competition/README.md`). `eval_competition/` reproduces these metrics and is the scorer used in our report. It re-scores the videos generated above with `face_score` (MTCNN + YuNet + FaceNet), `prompt_following_score` (SigLIP2), `image_quality_score` (MUSIQ) and `invisibility` (VMAF + SSIM):
```
pip install --no-deps facenet-pytorch==2.6.0
python eval_competition/run.py --method my_method
```
`invisibility` needs the [libvmaf](https://github.com/Netflix/vmaf/tree/master/libvmaf) CLI (`vmaf` on `PATH` or `VMAF_BIN=/path/to/vmaf`). The leaderboard's hidden prompts, temporal attacks and transfer models are not public, so these scores are a local estimate. See [`phase1/eval_competition/README.md`](phase1/eval_competition/README.md) and [`phase2/eval_competition/README.md`](phase2/eval_competition/README.md) for details.

### 4. Build & Check the Submission Archive (Optional)
```
cd data/protected/my_method/submission
python -m zipfile -c ../../../../submission.zip M003
cd ../../../..
python check_submission.py
```

## Acknowledgements
This repository is built on the official [ECCV 2026 LifeGenIP starting kit](https://github.com/tmllab/ECCV26_LifeGenIP_starting_kit). The data preparation, evaluation and packaging code under `data_preparation/`, `eval/`, `utils/` and `check_submission.py` comes from the kit unchanged, and the metric descriptions in `eval_competition/README.md` are the organizers'. The YuNet face detector is from [opencv_zoo](https://github.com/opencv/opencv_zoo), and the leaderboard metrics use [facenet-pytorch](https://github.com/timesler/facenet-pytorch), [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch), [SigLIP2](https://huggingface.co/google/siglip2-so400m-patch16-384) and [VMAF](https://github.com/Netflix/vmaf). We also use [Diffusers](https://github.com/huggingface/diffusers), [Wan2.2](https://github.com/Wan-Video/Wan2.2), [InsightFace](https://github.com/deepinsight/insightface), [Kornia](https://github.com/kornia/kornia) and [ai-toolkit](https://github.com/ostris/ai-toolkit). The videos come from the [MEAD dataset](https://wywu.github.io/projects/MEAD/MEAD.html); by downloading the data you agree to its [Terms of Use](https://github.com/uniBruce/Mead/blob/master/Terms%20of%20Use_%20Mead.pdf). Thanks to the organizers, authors and contributors for their great work!

This repository is a cleaned-up version of our code, prepared for reproducibility and simplicity. In this repository, all hyper-parameters are hard-coded, and the full method of each phase is consolidated into a single file (`phase*/protect/methods/my_method.py`). This is also the code we verified with the organizers, and it reproduces our perturbations (bit-accurate) under the same environment. Any factor that may affect reproduction (e.g., non-deterministic CUDA backward passes in the EOT interpolation) is documented in the corresponding phase README. Note that this repository does not reflect our original development code, and the consolidation may come at some cost to portability. We apologize for any inconvenience.

## Citation

The full technical report of this solution is *KVeil: First-Place Solution to ECCV 2026 LifeGenIP Challenge on Unlearnable Videos*. The method builds on our research paper *LETHE: Defending against Unauthorized Image-to-Video Generation via Value Diversity Collapse*, which presents the motivation, theoretical analysis and extensive experiments in detail. If you find this work helpful, please consider citing:

```bibtex
@article{li2026kveil,
  title   = {KVeil: First-Place Solution to ECCV 2026 LifeGenIP Challenge on Unlearnable Videos},
  author  = {Li, Boheng and Xing, Yueming and Li, Yiming and Zhang, Jie and Qiu, Han and Zhang, Tianwei},
  journal = {arXiv preprint},
  year    = {2026}
}

@article{li2026lethe,
  title   = {LETHE: Defending against Unauthorized Image-to-Video Generation via Value Diversity Collapse},
  author  = {Li, Boheng and Xing, Yueming and Li, Yiming and Zhang, Jie and Zhang, Tianwei},
  journal = {arXiv preprint},
  year    = {2026}
}
```

The development code of LETHE, with better readability and portability, will be released later. Stay tuned!

## License
This code is released under the [Apache License 2.0](LICENSE), following the starting kit.
