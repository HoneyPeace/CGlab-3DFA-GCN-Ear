# Main Computer 14ch Only Instructions 2026-05-23

본컴에서는 22ch/full-extension 교차 실험을 돌리지 않고, 14ch center-geometry 조합만 실행한다.

## 실행 명령

`Ear_3DFA-GCN` 폴더에서 아래 명령만 실행한다.

```powershell
powershell -ExecutionPolicy Bypass -File .\experiment_queues\run_main_14ch_centergeom_only_seed1.ps1
```

## 이 명령의 의미

- NPY raw input: `--in_channels 7`
- 7ch 구성: `XYZ 3ch + principal direction 3ch + curvature 1ch`
- PAConv 내부 edge feature: `--paconv_feature_mode center_geometry`
- DeepPA 내부 local feature: `--deeppa_feature_mode center_geometry`
- 내부 feature 채널: PAConv 14ch, DeepPA 14ch
- PAConv heatmap output: raw logits
- PAConv checkpoint/readout 기준: MDS
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
