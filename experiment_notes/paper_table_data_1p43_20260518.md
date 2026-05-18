# Paper Table Data Draft - 1.4317 Baseline

Date: 2026-05-18

Reference best result:

- Method: DeepPA Frozen HMR, heatmap-attention residual, feature+xyz pooling, 3 mm residual clamp
- Stage1 prior: `finetune_paconv_heatmap`
- Stage1 checkpoint: `results\S2G_PAconv_Refinement\FPS8192_sigma2.5_batch4_train200_finetune_paconv_heatmap_Stage1_PAConv_1\models\Single_PAConv_last.t7`
- Stage1 checkpoint SHA256: `9DC0C5AF99FB2DC235654D2297E0854E3887E14BF815C3CBCAC2EAFAA0B9F7F4`
- Main result: **1.4317 +/- 0.8755 mm**, 95%ile **1.8638 mm**
- Result file: `results\S2G_Frozen_HMAttnResidual_SigmoidPool500\FPS8192_sigma2.5_batch4_train200_pa163_hmattnres3mm_sigmoidpool_auxdrop30_trainhmplateau_topkcurv_500_Stage2_RAW_AuxDrop30_1\frozen_aux_drop_Results_ME1.4317.txt`

## Main Table Candidate

Use this table for the core paper claim. It avoids the unstable PAConv retraining rows and keeps the comparison centered on the fixed Stage1 prior.

| Method | Stage1 prior | Key setting | ME +/- STD (mm) | 95%ile ME (mm) | Delta vs 1.4317 | Heatmap cosine | mIoU | Surface normal (mm) | Suggested role |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Stage1 PAConv heatmap prior | `finetune_paconv_heatmap` | heatmap top-k | 1.6257 +/- 0.9020 | 2.0108 | +0.1940 | 79.94 | 57.04 | - | baseline |
| Frozen heatmap-only top-k | `finetune_paconv_heatmap` | heatmap-only, top-k | 1.4729 +/- 0.8815 | 2.0055 | +0.0412 | 89.44 | 66.37 | 0.2274 | strong baseline |
| DeepPA HMR full feature+xyz | `finetune_paconv_heatmap` | 3 mm clamp, train-HM plateau | **1.4317 +/- 0.8755** | **1.8638** | 0.0000 | 69.90 | 25.85 | 0.2173 | main result |

Recommended wording:

- Compared with the fixed PAConv prior, DeepPA HMR reduces ME from **1.6257 mm** to **1.4317 mm**.
- Compared with the heatmap-only frozen baseline, DeepPA HMR reduces ME from **1.4729 mm** to **1.4317 mm**.
- Do not claim that HMR improves heatmap quality. The heatmap-only row has higher cosine/mIoU; the HMR gain is in final coordinate accuracy.

## Ablation Table Candidate

Use this table to show what each modeling choice changes. Keep the language modest because several differences are small.

| Variant | ME +/- STD (mm) | 95%ile ME (mm) | Delta vs 1.4317 | Surface normal (mm) | Interpretation |
|---|---:|---:|---:|---:|---|
| Full feature+xyz residual, train-HM plateau, 3 mm | **1.4317 +/- 0.8755** | **1.8638** | 0.0000 | 0.2173 | best fixed-prior result |
| Feature-only residual, train-HM plateau, 3 mm | 1.4376 +/- 0.8588 | 2.0002 | +0.0059 | - | close to full model; xyz effect is marginal |
| No plateau / immediate geometry, 3 mm | 1.4428 +/- 0.8596 | 1.8860 | +0.0111 | - | plateau helps modestly with this fixed prior |
| P2P-only surface, 3 mm | 1.4609 +/- 0.8657 | 1.9451 | +0.0292 | 0.2203 | surface variant is weaker than top-k diagnostics setting |
| Batch size 8, 3 mm | 1.4830 +/- 0.8813 | 1.8586 | +0.0513 | 0.2120 | batch change worsened ME |

## Clamp Sweep Table

Use this as internal/appendix unless the paper needs the exact 3 mm justification. The trend is not monotonic, so it should not be framed as "larger/smaller clamp is consistently better."

| Clamp | ME +/- STD (mm) | 95%ile ME (mm) | Delta vs 1.4317 | Surface normal (mm) | Note |
|---:|---:|---:|---:|---:|---|
| 0 mm / no clamp | 1.4511 +/- 0.8706 | 1.8923 | +0.0194 | 0.2260 | weaker than 3 mm |
| 1 mm | 1.4351 +/- 0.8799 | 1.9597 | +0.0034 | 0.2152 | nearly tied with 3 mm |
| 2 mm | 1.4777 +/- 0.8970 | 1.8941 | +0.0460 | 0.2180 | worse |
| 3 mm | **1.4317 +/- 0.8755** | **1.8638** | 0.0000 | 0.2173 | best ME in sweep |
| 4 mm | 1.4796 +/- 0.8943 | 2.1070 | +0.0479 | 0.2205 | worse |
| 5 mm | 1.4734 +/- 0.8934 | 1.9220 | +0.0417 | 0.2254 | worse |

Suggested interpretation:

- The 3 mm clamp is an empirical stabilizer selected on the fixed-prior ablation.
- The sweep supports using a bounded residual, but it does not support a monotonic story about clamp size.
- If space is tight, include only no-clamp vs 3 mm in the main paper and move the full sweep to supplementary material.

## Reproducibility / Robustness Table

Keep this separate from the main claim. These rows are useful for internal discussion, rebuttal prep, or limitation wording.

| Condition | Stage1 prior | ME +/- STD (mm) | 95%ile ME (mm) | Delta vs 1.4317 | Interpretation |
|---|---|---:|---:|---:|---|
| Current-code fixed-prior repro | `finetune_paconv_heatmap` | 1.4755 +/- 0.8743 | 1.9206 | +0.0438 | same prior, current code path; old-commit repro still running |
| PAConv rerun prior + HMR plateau | `paconv_heat_final_split_rerun_20260516` | 1.5195 +/- 0.9350 | 1.9930 | +0.0878 | downstream HMR sensitive to Stage1 prior |
| PAConv rerun prior + HMR no plateau | `paconv_heat_final_split_rerun_20260516` | 1.4788 +/- 0.9411 | 1.9906 | +0.0471 | no-plateau beats plateau under rerun prior |

Recommended handling:

- Main paper: report fixed-prior results and provide the exact released PAConv checkpoint.
- Limitation/reproducibility note: PAConv from-scratch training has measurable variance; downstream HMR should be reproduced from the released Stage1 prior.
- Avoid claiming that train-HM plateau or 3 mm clamp is universally beneficial across all newly trained PAConv priors.

## Stage1 Reference Rows

These rows explain why checkpoint identity matters.

| Stage1 PAConv condition | ME +/- STD (mm) | 95%ile ME (mm) | Heatmap cosine | mIoU | Note |
|---|---:|---:|---:|---:|---|
| `finetune_paconv_heatmap` prior used for 1.4317 | 1.6257 +/- 0.9020 | 2.0108 | 79.94 | 57.04 | released/fixed prior for main HMR |
| `paconv_heat_final_split` | 1.5846 +/- 0.8847 | 2.0031 | 80.05 | 56.97 | better PAConv-only ME, not the 1.4317 prior |
| `paconv_heat_final_split_rerun_20260516` | 1.6228 +/- 0.9027 | 2.0489 | 80.04 | 56.42 | similar heatmap metrics, different downstream HMR behavior |

## Paper Table Recommendation

1. Main quantitative table:
   - Stage1 PAConv prior
   - Frozen heatmap-only top-k
   - DeepPA HMR full feature+xyz

2. Ablation table:
   - Full feature+xyz
   - Feature-only
   - No-plateau/immediate geometry
   - P2P-only surface

3. Supplementary/internal table:
   - Clamp sweep 0/1/2/3/4/5 mm
   - PAConv rerun robustness

One-line paper-safe claim:

> Using the released fixed PAConv heatmap prior, the proposed heatmap-guided residual refinement improves final coordinate accuracy from 1.4729 mm for frozen heatmap-only decoding to 1.4317 mm, while retaining 100% SR@5 mm and SR@10 mm on the test split.
