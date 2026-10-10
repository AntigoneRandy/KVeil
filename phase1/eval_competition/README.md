# Phase 1: Evaluation

In this phase the protection is applied to a single video clip of a single identity. The threat scenario is reference-based customization: an attacker samples a random frame from the protected video clip and feeds it to an Image-to-video (I2V) model with a text prompt to generate a new video.

**Evaluation Tracks**: In this phase, we evaluate each submission under two tracks:

- **Effectiveness**: the protected video clip is evaluated as-is.
- **Robustness**: the protected video clip is first processed by a temporal attack before evaluation.



## Table of Contents

- [Effectiveness](#effectiveness)
- [Robustness](#robustness)
- [Hidden Metrics](#hidden-metrics)



## Effectiveness

Phase 1 evaluates effectiveness using [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers).

## Robustness

Phase 1 evaluates robustness using [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers). Each hidden temporal attack is applied to the protected video before I2V.

The attacks applied in each robustness setting are listed below:

- `robustness_attack1`: Neighbor-frame interpolation.
- `robustness_attack2`: 7-frame temporal averaging.
- `robustness_attack3`: H.264 re-compression.



## Hidden Metrics

Each score is computed per video using the definitions below. Scores from different videos are averaged to obtain the final score for each evaluation setting.


| Score                    | Score Computation                                                                                                                          | Metric / Model                                                                                                                                                                                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `invisibility`           | `(VMAF + 100 × SSIM) / 2`                                                                                                                  | [VMAF](https://github.com/Netflix/vmaf) and SSIM                                                                                                                                                                                                            |
| `face_score`             | Average over generated frames. A frame scores `100` if either detector fails; otherwise, `100 × max(0, 1 - FaceNet cosine similarity)`.    | Face detection: [MTCNN](https://github.com/timesler/facenet-pytorch) and [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet). Identity embedding: [FaceNet InceptionResnetV1](https://github.com/timesler/facenet-pytorch). |
| `prompt_following_score` | `100 × (0.2 - clamp(SigLIP2 similarity, 0, 0.2)) / 0.2`, where `0.2` is an empirically chosen upper bound for prompt-following similarity. | [SigLIP2 So400m Patch16-384](https://huggingface.co/google/siglip2-so400m-patch16-384)                                                                                                                                                                      |
| `image_quality_score`    | `clamp(100 - MUSIQ, 0, 100)`                                                                                                               | [MUSIQ](https://github.com/chaofengc/IQA-PyTorch)                                                                                                                                                                                                           |


For each track, `track score = average(face_score, prompt_following_score, image_quality_score)`.

The final score is computed as:

```text
score = (0.5 * invisibility + 4 * HarmonicMean(effectiveness, robustness_attack1, robustness_attack2, robustness_attack3)) / 4.5
```




## Reproduction of the Leaderboard Metrics

The section above is the organizers' description of the leaderboard metrics, from the [starting kit](https://github.com/tmllab/ECCV26_LifeGenIP_starting_kit). `scorer.py` reproduces these metrics following that description, and is the scorer used for the experiments in our report.

- `scorer.py`: `face_score`, `prompt_following_score`, `image_quality_score`, track score and `invisibility`, following the table above.
- `run.py`: scores the videos that the kit's `eval/run.py` generated under `data/generated/<method>/` and writes `data/results_competition_<method>.json`.
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
