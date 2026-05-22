# Main Computer 14ch Only Instructions 2026-05-23

본컴에서는 22ch/full-extension 교차 실험을 돌리지 않고, 14ch center-geometry 조합만 실행한다.

## 실행 순서

`Ear_3DFA-GCN` 폴더에서 먼저 PAConv-only를 학습/평가한다.

```powershell
powershell -ExecutionPolicy Bypass -File .\experiment_queues\run_mainpaconv_raw7_centergeom_seed1.ps1
```

이 명령이 끝나면 최신 `Single_PAConv_last.t7`를 DeepPA Stage1 prior로 불러와서 후속 DeepPA 큐를 실행한다.

```powershell
powershell -ExecutionPolicy Bypass -File .\experiment_queues\run_main14_deeppa_followup_queue_seed1.ps1
```

## PAConv 이후 DeepPA 후속 예약 큐

PAConv stage1을 14ch center-geometry로 맞춘 뒤 DeepPA에서 볼 후속 실험 5개도 예약 파일로 준비해 두었다.
이 큐는 PAConv를 새로 학습하지 않고, 방금 PAConv-only 명령이 만든 최신 `Single_PAConv_last.t7`를 모든 variant의 Stage1 checkpoint로 공유한다.

큐에 들어간 DeepPA 후속 실험은 아래 5개다.

| 순서 | variant | 확인하려는 것 |
|---:|---|---|
| 1 | `base_auxdrop_plateau_fps_hmr2_seed1` | 기본값: aux 30epoch drop + plateau + FPS + 2mm HMR |
| 2 | `auxfixed_plateau_fps_hmr2_seed1` | aux를 0.1 고정으로 유지하면 안정성이 좋아지는지 |
| 3 | `noaux_plateau_fps_hmr2_seed1` | aux heatmap을 빼도 prior/residual만으로 충분한지 |
| 4 | `auxdrop_plateau_grid_hmr2_seed1` | DeepPA stage downsampling을 FPS 대신 grid로 바꾸면 좋아지는지 |
| 5 | `auxdrop_plateau_fps_hmr1p5_seed1` | HMR boundary를 2.0mm에서 1.5mm로 줄이면 좋아지는지 |

주의: 이 큐는 14ch center-geometry 전용이다. full-extension 22ch 교차 실험은 본컴 기본 지시에서 제외한다.

## 평가 실행 원칙

학습 하나가 끝날 때마다 같은 설정으로 평가까지 이어서 실행한다.

- PAConv 단독 실행 파일은 `run.py`를 통해 학습 후 `eval.py`를 자동 실행한다.
- PAConv+DeepPA 실행 파일은 `run_frozen.py`를 통해 Stage1 PAConv 평가 후, 각 DeepPA variant 학습이 끝날 때마다 `eval.py`를 자동 실행한다.
- 각 평가에는 학습 때 쓴 `exp_name`, `user_tag`, `feature_mode`, `eval_heatmap_coord_method`, `need_resample=False`, seed 설정이 그대로 전달된다.
- 평가 산출물이 이미 있으면 기존 결과를 재사용하고, 없으면 새로 평가한다.
- 평가 실패 시 해당 variant의 `.err.log`를 먼저 확인하고 같은 설정으로만 재평가한다.

## 이 명령의 의미

- NPY raw input: `--in_channels 7`
- 7ch 구성: `XYZ 3ch + principal direction 3ch + curvature 1ch`
- PAConv 내부 edge feature: `--paconv_feature_mode center_geometry`
- DeepPA 내부 local feature: `--deeppa_feature_mode center_geometry`
- 내부 feature 채널: PAConv 14ch, DeepPA 14ch
- PAConv heatmap output: raw logits
- PAConv checkpoint/readout 기준: MDS, DeepPA Stage1은 PAConv-only 최신 `Single_PAConv_last.t7`
- DeepPA validation/eval ME: 원좌표 복원 후 Euclidean distance
- `Train_mm`: 증강된 학습 좌표계 기준 Euclidean distance
- NPY cache: `--need_resample False`
- Seed: `--seed 1 --dataset_seed 1`

## 실행하지 말 것

본컴 기본 실험에서는 아래 full-extension 조합을 실행하지 않는다.

```powershell
.\experiment_queues\run_paconvfull_deeppacenter_seed1.ps1
.\experiment_queues\run_paconvcenter_deeppafull_seed1.ps1
.\experiment_queues\run_paconvfull_deeppafull_seed1.ps1
```

## 결과 정리

실험이 끝나면 Notion `논문 정리 모음`의 원본 PAConv/DeepPA 후속 정리 페이지에 다음 항목을 적는다.

- 실험명: PAConv center-geometry 14ch + DeepPA center-geometry 14ch
- result folder
- best Val ME
- final MDS ME ± STD
- final TopK ME ± STD
- 해석: 7ch raw input을 쓰지만 내부 geometry는 14ch center-only 구조로 PAConv와 DeepPA를 맞춘 실험
