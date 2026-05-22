# Main PAConv / DeepPA follow-up 실행 계획 2026-05-22

이 문서는 메인컴에서 `Ear_3DFA-GCN` 폴더만 보고 바로 실행할 수 있도록 정리한 PAConv/DeepPA 후속 실험 메모이다.

## 현재 결론

- 우선 seed 고정으로 간다: `--seed 1 --dataset_seed 1`.
- NPY cache는 유지한다: `--need_resample False`.
- PAConv 최종 heatmap 출력은 기본 raw logits로 둔다: `--paconv_heatmap_activation_mode raw`.
- ScoreNet 내부 softmax는 PAConv 원래 동적 커널 가중치용이므로 유지한다: `--calc_scores softmax`.
- PAConv head는 재현 편의를 위해 `1344 -> 256 -> 256 -> 128 -> 36`으로 간다.
- PAConv가 DeepPA prior로 줄 latent는 `prior_hints = 1344ch x_concat` 그대로 둔다.
- DeepPA 입력 채널 구조는 일단 바꾸지 않는다.
- DeepPA/PAConv validation은 eval과 같은 방식으로 원좌표 복원 후 Euclidean ME를 쓴다.
- PAConv checkpoint 선택/비교 기준은 MDS로 두고, 최종 eval에서는 MDS와 TopK를 둘 다 기록한다.

## PAConv 입력 구조

### 3ch raw PAConv

입력 NPY: XYZ 3채널.

KNN과 distance는 XYZ 기준이다. 현재 main PAConv edge feature는 아래 10채널이다.

```text
[neighbor_xyz - center_xyz, neighbor_xyz, center_xyz, xyz_distance]
= [dx, dy, dz, nx, ny, nz, cx, cy, cz, dist_xyz]
```

### 7ch center-geometry PAConv

입력 NPY: 7채널.

```text
[x, y, z, principal_dir_x, principal_dir_y, principal_dir_z, curvature]
```

현재 main PAConv는 기존 original-clone 7ch처럼 모든 7채널의 neighbor/center/delta/distance를 섞지 않는다. XYZ는 관계/위상 계산에 쓰고, principal direction + curvature는 중심점 geometry hint로만 붙인다.

```text
[neighbor_xyz - center_xyz, neighbor_xyz, center_xyz, xyz_distance, center_principal_dir_3ch, center_curvature_1ch]
= 14ch
```

즉 7ch 입력이지만 "PAConv식 full 7ch 확장"이 아니라 "XYZ relation + center geometry" 설계이다.

## 이미 확인된 original-clone PAConv 결과

| 실험 | 입력/구조 | checkpoint 기준 | final MDS ME±STD | final TopK ME±STD | 비고 |
|---|---|---:|---:|---:|---|
| original-clone PAConv xyz MDS | XYZ only, original GitHub clone | MDS | 1.5042 ± 1.3422 | 1.5056 ± 1.3392 | overwritten NPY cache |
| original-clone PAConv xyz TopK | XYZ only, original GitHub clone | TopK | 약 1.5110 ± 1.3251 | 약 1.5096 ± 1.3147 | MDS와 거의 동일 |
| original-clone PAConv 7ch MDS | PAConv-style full 7ch extension | MDS | 진행 완료, final 정리 필요 | 진행 완료, final 정리 필요 | best Val MDS Ep495 1.5521 ± 1.2192 |
| original-clone PAConv 7ch TopK | PAConv-style full 7ch extension | TopK | 진행/정리 필요 | 진행/정리 필요 | 현재 큐에서 이어서 진행 |

주의: 위 7ch original-clone 결과는 이 문서의 main 7ch center-geometry 설계와 다르다.

## 메인컴/서브컴 실행 스크립트

두 실험 모두 `Ear_3DFA-GCN` 폴더에서 실행한다.

### 1. 3ch raw PAConv

서브컴 또는 비교용 컴퓨터에서 실행 권장.

```powershell
powershell -ExecutionPolicy Bypass -File .\experiment_queues\run_mainpaconv_raw3_seed1.ps1
```

결과 root:

```text
..\results\MainPAConv_DeepPAReady\S2G_MainPAConv_Raw3_XYZRel_MDS_Seed1
```

### 2. 7ch center-geometry raw PAConv

메인컴 우선 실행 권장.

```powershell
powershell -ExecutionPolicy Bypass -File .\experiment_queues\run_mainpaconv_raw7_centergeom_seed1.ps1
```

결과 root:

```text
..\results\MainPAConv_DeepPAReady\S2G_MainPAConv_Raw7_CenterGeom_MDS_Seed1
```

두 스크립트는 공통 runner `experiment_queues\run_mainpaconv_seed1.ps1`을 사용한다. 로그는 `debug_outputs\run_mainpaconv_*.out.log`, `.err.log`, `.summary.log`에 남는다.

## 실행 기본 옵션

스크립트가 공통으로 쓰는 핵심 옵션:

```text
--model paconv_heat
--need_resample False
--seed 1
--dataset_seed 1
--epochs 500
--train_len 200
--batch_size 4
--test_batch_size 1
--num_points 8192
--sigma 2.5
--k 30
--regression_point_num 10
--latent_injection_type raw
--paconv_heatmap_activation_mode raw
--eval_heatmap_coord_method mds
--heatmap_loss_mode adaptive_wing
--calc_scores softmax
```

## 코드 변경 요약

- `My_args.py`
  - PAConv heatmap activation 기본값을 `raw`로 둔다.
- `PAConv_model.py`
  - PAConv head를 `1344 -> 256 -> 256 -> 128 -> 36`으로 정리한다.
  - PAConv 최종 heatmap softmax를 사용할 경우 포인트 축(`dim=2`)으로 적용한다.
  - 기본값은 softmax/sigmoid가 아니라 raw logits이다.
- `PAConv/util/PAConv_util.py`
  - 3ch는 10ch edge feature를 만든다.
  - 7ch는 XYZ relation + center principal direction/curvature 구조의 14ch edge feature를 만든다.
- `train.py`
  - validation ME를 eval과 동일하게 원좌표 복원 후 Euclidean distance 평균으로 계산한다.

이 변경 때문에 기존 L1-scaled validation 숫자와 새 validation 숫자는 직접 비교하지 않는다. 새 validation은 final eval ME와 같은 단위/방식이다.

## PAConv 완료 후 DeepPA

PAConv 결과가 나온 뒤 final MDS/TopK가 좋은 checkpoint를 Stage1 prior로 고정하고 DeepPA frozen run을 실행한다.

기본 방향:

- `prior_hints = 1344ch x_concat`
- DeepPA 입력 채널은 현 구조 유지
- `--need_resample False`
- seed 고정
- HMR residual boundary는 우선 기존 best 계열인 2mm를 기준으로 둔다.
- aux heatmap은 유지하고, HDS/grid sampling 비교는 후순위 ablation으로 분리한다.

Stage2 실행 명령은 선택된 PAConv 결과 folder와 checkpoint가 확정된 뒤 별도 스크립트로 고정한다.

## Notion 정리 지시

각 실험이 끝나면 `논문 정리 모음`의 original PAConv 재현/후속 페이지에 표로 정리한다.

최소 컬럼:

```text
experiment
code base
input channels/features
PAConv edge feature rule
heatmap activation
checkpoint readout
best validation
final MDS ME±STD
final TopK ME±STD
result folder
interpretation
```

반드시 따로 구분할 행:

- original-clone PAConv xyz MDS
- original-clone PAConv xyz TopK
- original-clone PAConv 7ch PAConv-style full extension MDS
- original-clone PAConv 7ch PAConv-style full extension TopK
- main PAConv raw 3ch XYZ relation MDS
- main PAConv raw 7ch center-geometry MDS
- 이후 선택된 PAConv checkpoint 기반 frozen DeepPA 결과

해석에는 다음을 꼭 적는다.

- original-clone 7ch는 main center-geometry 설계와 다르다.
- MDS와 TopK readout은 현재 PAConv에서는 거의 같은 수준으로 보인다.
- 이전 validation과 eval 차이는 validation metric 방식 차이일 가능성이 컸고, 이번 코드에서는 val/eval을 Euclidean ME로 통일했다.
- PAConv 최종 softmax는 기본 제거(raw)했으며, ScoreNet softmax와는 역할이 다르다.

## 다음 의사결정

1. main PAConv raw 3ch vs raw 7ch 결과 비교
2. 더 좋은 PAConv checkpoint로 frozen DeepPA 실행
3. raw가 애매하면 sigmoid PAConv ablation 추가
4. DeepPA grid sampling 복구와 HDS/aux 대체성 비교는 그 다음 단계
5. 가장 좋은 후보에서 seed 1/2/3 반복으로 안정성 확인
