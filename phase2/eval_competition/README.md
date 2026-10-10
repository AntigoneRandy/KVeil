# Phase 2: Evaluation

In this phase the protection is applied to a set of video clips of a single identity. The threat scenario covers both *reference-based* and *tuning-based* customization.

In the *reference-based* setting, an attacker samples a random frame from video clips and uses it as the reference input to an image-to-video (I2V) model. In the *tuning-based* setting, the attacker fine-tunes a text-to-video (T2V) model on the entire video set and then generates new videos.

**Evaluation Tracks**: In this phase, each submission is evaluated under three tracks in both the reference-based and tuning-based customization settings:

- **Effectiveness**: the protected video set is evaluated as-is using the known evaluation model.
- **Robustness**: the protected video set is first processed by a temporal attack before evaluation.
- **Transferability**: the protected video set is evaluated using additional hidden video generation models.



## Table of Contents

- [Effectiveness](#effectiveness)
- [Robustness](#robustness)
- [Transferability](#transferability)
- [Hidden Metrics](#hidden-metrics)

## Effectiveness

Phase 2 evaluates effectiveness against [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) under both customization settings. Specifically, the tuning-based setting uses LoRA fine-tuning through [ai-toolkit](https://github.com/ostris/ai-toolkit).

## Robustness

Robustness is evaluated using [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) under both customization settings. 

The attacks applied in each robustness setting are listed below:

- **Reference-based setting**
  - `robustness_ref_attack1`: 7-frame temporal averaging, followed by H.264 re-compression.
  - `robustness_ref_attack2`: [VideoPure](https://github.com/deep-kaixun/VideoPure) purification.
- **Tuning-based setting**
  - `robustness_tune_attack1`: 5-frame temporal averaging, followed by temporal reversal.
  - `robustness_tune_attack2`: H.264 re-compression.



## Transferability

Transferability evaluates whether a protection method generalizes beyond the known [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) model. Two additional models are evaluated for each customization setting.

- **Reference-based setting**
  - `transferability_ref_model1`: Uses [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5), with I2V inference through [musubi-tuner](https://github.com/kohya-ss/musubi-tuner).
  - `transferability_ref_model2`: Uses [Kling 3.0](https://kling.ai/), with standard image-to-video generation through the Kling API.
- **Tuning-based setting**
  - `transferability_tune_model1`: Uses [LTX-2](https://huggingface.co/Lightricks/LTX-2), with LoRA training and inference through [official LTX-2 repository](https://github.com/Lightricks/LTX-2).
  - `transferability_tune_model2`: Uses [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5), with LoRA training and inference through [musubi-tuner](https://github.com/kohya-ss/musubi-tuner).



## Hidden Metrics

Each score is computed per video using the definitions below. Scores from different videos are averaged to obtain the final score for each evaluation setting.


| Score                    | Score Computation                                                                                                                          | Metric / Model                                                                                                                                                                                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `invisibility`           | `(VMAF + 100 × SSIM) / 2`                                                                                                                  | [VMAF](https://github.com/Netflix/vmaf) and SSIM                                                                                                                                                                                                            |
| `face_score`             | Average over generated frames. A frame scores `100` if either detector fails; otherwise, `100 × max(0, 1 - FaceNet cosine similarity)`.    | Face detection: [MTCNN](https://github.com/timesler/facenet-pytorch) and [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet). Identity embedding: [FaceNet InceptionResnetV1](https://github.com/timesler/facenet-pytorch). |
| `prompt_following_score` | `100 × (0.2 - clamp(SigLIP2 similarity, 0, 0.2)) / 0.2`, where `0.2` is an empirically chosen upper bound for prompt-following similarity. | [SigLIP2 So400m Patch16-384](https://huggingface.co/google/siglip2-so400m-patch16-384)                                                                                                                                                                      |
| `image_quality_score`    | `clamp(100 - MUSIQ, 0, 100)`                                                                                                               | [MUSIQ](https://github.com/chaofengc/IQA-PyTorch)                                                                                                                                                                                |


For each track, `track score = average(face_score, prompt_following_score, image_quality_score)`.

The final score is computed as:

```text
score = (
    0.5 * invisibility
    + 5 * HarmonicMean(
        effectiveness_ref,
        robustness_ref_attack1,
        robustness_ref_attack2,
        transferability_ref_model1,
        transferability_ref_model2,
        effectiveness_tune,
        robustness_tune_attack1,
        robustness_tune_attack2,
        transferability_tune_model1,
        transferability_tune_model2
    )
) / 5.5
```




## Reproduction of the Leaderboard Metrics

The section above is the organizers' description of the leaderboard metrics, from the [starting kit](https://github.com/tmllab/ECCV26_LifeGenIP_starting_kit). `scorer.py` reproduces these metrics following that description, and is the scorer used for the experiments in our report.

- `scorer.py`: `face_score`, `prompt_following_score`, `image_quality_score`, track score and `invisibility`, following the table above.
- `run.py`: scores the videos that the kit's `eval/run.py` generated under `data/generated/<method>/` and writes `data/results/<method>/results_competition.json`.
- `models/face_detection_yunet_2023mar.onnx`: the YuNet detector from [opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT License), md5 `4ae92eeb150c82ce15ac80738b3b8167`.

Usage:
```
python eval/run.py --method my_method                    # kit: generate videos
python eval_competition/run.py --method my_method        # re-score with the metrics above
```

Extra dependencies, on top of the kit's evaluation environment:
```
pip install --no-deps facenet-pytorch==2.6.0             # MTCNN + FaceNet; --no-deps keeps your torch
```
`invisibility` also needs the Netflix [libvmaf](https://github.com/Netflix/vmaf/tree/master/libvmaf) command-line tool (`vmaf`, model `vmaf_v0.6.1`) and `ffmpeg`. Put `vmaf` on `PATH`, or set `VMAF_BIN=/path/to/vmaf`. Without it, VMAF and `invisibility` are reported as `n/a` and the other scores are unaffected.

Choices the official description leaves open, and how we resolved them:
1. **Identity reference.** The reference embedding is FaceNet on the *original* (unprotected) clip's reference frame, as in the kit's `eval/run.py`.
2. **Aggregation.** `image_quality` and `prompt_following` are scored per frame and then averaged, like `face_score`. The other order (average first, then apply the formula) is stored under `alt` in the output JSON.
3. **Face score cap.** A single frame's face score is capped at 100 (a negative FaceNet cosine would exceed it). The uncapped mean is stored as `face_uncapped`.
4. **SSIM / VMAF.** SSIM is pyiqa's `ssim`, averaged over frames. VMAF is the pooled mean of `vmaf_v0.6.1` on yuv420p.

> ⚠️ Only the prompts, reference frame, models and temporal attacks enabled in the kit's `eval/config.yaml` are scored. The leaderboard's hidden prompts, frame sampling, attacks and transfer models are not public, so these numbers are a local estimate, not the leaderboard score.
