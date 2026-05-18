# Final Setting, Architecture, and Run Guide

Date: 2026-05-19
Project: Ear 3DFA-GCN / PAConv-prior Frozen HMR
Main model family: Frozen PAConv prior + DeepPA heatmap-attention residual coordinate readout

This note summarizes the current paper-facing configuration, the actual model architecture implemented in the code, the training/evaluation flow, and the latest seed-controlled reproduction run. It is written so that another writing agent or researcher can understand the method without digging through the code first.

---

## 1. Executive Summary

### Paper-facing final method

The recommended main setting is:

- Stage 1: frozen PAConv heatmap prior.
- Stage 2: DeepPA refinement network.
- Prior injection: raw PAConv latent feature injection.
- Decoder fusion: progressive half fusion with final 320-channel decoder feature.
- Coordinate readout: heatmap-attention pooling plus residual coordinate correction.
- Residual bound: 3 mm maximum correction.
- Heatmap supervision: main heatmap plus stage-wise auxiliary heatmaps.
- Aux schedule: auxiliary heatmap loss linearly dropped over 30 epochs.
- Loss schedule: heatmap-only warm-up until train heatmap plateau, then geometry losses are enabled.
- Geometry losses: coordinate focal-L1, top-k curvature-aware surface loss, and coordinate structural loss.

The key scientific idea is:

> Use PAConv as a frozen heatmap/latent prior, use DeepPA to refine the dense feature representation, estimate a probability heatmap over points, and regress final landmark coordinates by bounded residual correction around the heatmap-attention pooled coordinate.

### Important distinction

There are two different notions of reproducibility:

- Checkpoint evaluation reproducibility: loading a saved HMR checkpoint and evaluating it again.
- From-scratch Stage2 training reproducibility: reinitializing the Stage2 HMR/DeepPA model and training again.

The saved best checkpoint remains the strongest paper result:

```text
Existing best HMR checkpoint eval:
Average ME = 1.4317 +/- 0.8755 mm
95%ile ME = 1.8638 mm
```

The latest seed-controlled Stage2 retraining using the same method and same PAConv prior produced:

```text
Seed-fixed Stage2 retraining, seed=1:
Average ME = 1.4737 +/- 0.8780 mm
95%ile ME = 1.8958 mm
```

This means the checkpoint result is evaluable, but Stage2 from-scratch training still shows trajectory variance.

---

## 2. Data Representation

### Input tensor

The model uses a sampled point cloud with:

```text
N = 8192 points
C = 7 channels
```

Each point feature is:

$$
f_i = [x_i, y_i, z_i, d^x_i, d^y_i, d^z_i, \kappa_i] \in \mathbb{R}^{7}
$$

The full input is:

$$
X = [f_1, f_2, \ldots, f_N] \in \mathbb{R}^{N \times 7}
$$

During model forward pass this is transposed to:

$$
X^\top \in \mathbb{R}^{7 \times N}
$$

In code:

```text
shape_*.npy channel order:
0: x
1: y
2: z
3: principal direction x
4: principal direction y
5: principal direction z
6: curvature
```

### Curvature feature

The curvature scalar is computed from a local kNN patch covariance. Let the local covariance eigenvalues be:

$$
\lambda_1 \le \lambda_2 \le \lambda_3
$$

The curvature descriptor is:

$$
\kappa_i = \frac{\lambda_1}{\lambda_1 + \lambda_2 + \lambda_3 + \epsilon}
$$

In code this is produced in `util.py` by `compute_geometric_features_7ch`, with:

```text
k = 15
principal_dir = eigvec[..., 2]
curvature = eigval[..., 0] / sum(eigval)
```

### Normalization behavior

Only XYZ coordinates are centered and scaled during training/evaluation normalization. The geometric channels are preserved except for scale/translate augmentation of direction vectors.

In `augmentations.py`:

```text
XYZ:
  centered by centroid
  divided by max radius m

direction/curvature:
  direction is normalized under scale augmentation
  curvature is kept as scalar feature
```

Landmark coordinates are normalized with the same centroid and scale as XYZ, and evaluation denormalizes predicted coordinates back to millimeters.

---

## 3. Stage 1: PAConv Prior

### Role

PAConv is used as a frozen prior. It provides:

1. A dense latent feature prior.
2. A Stage1 heatmap anchor.

The Stage1 PAConv parameters are not updated in the frozen HMR setting.

### PAConv checkpoint

The paper-facing Stage1 prior is:

```text
C:\Users\CGlab\Desktop\Earlandmark\results\S2G_PAconv_Refinement\FPS8192_sigma2.5_batch4_train200_finetune_paconv_heatmap_Stage1_PAConv_1\models\Single_PAConv_last.t7
```

SHA256:

```text
9DC0C5AF99FB2DC235654D2297E0854E3887E14BF815C3CBCAC2EAFAA0B9F7F4
```

This is bridged into:

```text
C:\Users\CGlab\Desktop\Earlandmark\results\PAConv_Pretrained\models\Single_PAConv_last.t7
```

### PAConv latent prior shape

In `PAConv_model.py`, raw PAConv prior is:

$$
H_{\text{PA}} \in \mathbb{R}^{1344 \times N}
$$

The 1344 channels come from:

```text
local PAConv features: 64 x 5 = 320
global pooled feature: 1024
total: 320 + 1024 = 1344
```

In code:

```python
xx = torch.cat((x1, x2, x3, x4, x5), dim=1)
xc = F.adaptive_max_pool1d(xc, 1).view(B, -1)
cls = xc.view(B, 1024, 1).repeat(1, 1, N)
x_concat = torch.cat((xx, cls), dim=1)
prior_hints = x_concat
```

---

## 4. Stage 2: DeepPA Prior-Guided Refinement

### Backbone

Stage2 uses `DeepPA_Wrapper` and `DeepPA_semseg`.

Default multi-stage settings:

```text
depths  = [20, 20, 60, 20]
dims    = [64, 128, 256, 512]
npoints = [2048, 512, 128, 32]
ks      = [20, 20, 20, 20]
```

The point hierarchy is:

$$
8192 \rightarrow 2048 \rightarrow 512 \rightarrow 128 \rightarrow 32
$$

### Prior injection

Current setting:

```text
latent_injection_type = raw
fusion_residual_base = prior
use_interaction_fusion = True
```

For raw injection, the PAConv prior is projected to the current stage channel dimension:

$$
P_l = \phi_l(H_{\text{PA}})
$$

where:

$$
\phi_l : \mathbb{R}^{1344} \rightarrow \mathbb{R}^{C_l}
$$

The implemented fusion is:

$$
F_l^{\text{fused}} = P_l + \psi_l([F_l, P_l])
$$

where:

- $F_l$ is the DeepPA feature at stage $l$.
- $P_l$ is the projected PAConv prior.
- $\psi_l$ is the interaction fusion MLP.
- The residual base is `prior`, so the fused feature is prior-centered.

This is important: `latent_injection_type raw` does not mean "no residual". It means the full 1344-channel PAConv latent prior is injected after projection. The actual fusion still uses a residual-style interaction fusion.

### Decoder fusion

Current setting:

```text
decoder_fusion = prog_half_final320
```

This progressively merges decoder features and produces a final dense decoder feature with:

$$
F_{\text{dec}} \in \mathbb{R}^{320 \times N}
$$

Then XYZ is concatenated:

$$
F_{\text{head-in}} = [F_{\text{dec}}, X_{xyz}] \in \mathbb{R}^{323 \times N}
$$

This is why the final head input is 323 channels:

```text
320 decoder channels + 3 XYZ channels = 323
```

---

## 5. Heatmap and Coordinate Head

### Head structure

In `DeepPA_model.py`, the head is:

```text
head_conv:
  Conv1d(323 -> 1024)
  BatchNorm
  ReLU
  Conv1d(1024 -> 512)
  BatchNorm
  ReLU

main_heatmap_head:
  Conv1d(512 -> landmark_num)
```

With 36 landmarks:

$$
Z \in \mathbb{R}^{36 \times N}
$$

The heatmap probability is:

$$
P = \sigma(Z)
$$

where $\sigma$ is sigmoid.

### Heatmap-attention residual coordinate readout

Current setting:

```text
train_coord_readout = heatmap_attn_residual
hm_attn_residual_max_mm = 3.0
```

For each landmark $j$, attention weights are:

$$
a_{j,i} = \frac{\sigma(z_{j,i})}{\sum_{i=1}^{N}\sigma(z_{j,i}) + \epsilon}
$$

The heatmap-pooled coordinate is:

$$
\bar{x}_j = \sum_{i=1}^{N} a_{j,i} x_i
$$

The heatmap-pooled feature is:

$$
\bar{f}_j = \sum_{i=1}^{N} a_{j,i} f_i^{512}
$$

The residual MLP input is:

$$
r_j^{\text{in}} = [\bar{f}_j, \bar{x}_j] \in \mathbb{R}^{515}
$$

because:

```text
512 pooled feature channels + 3 pooled XYZ channels = 515
```

The residual is:

$$
\Delta x_j = \text{MLP}_{515 \rightarrow 256 \rightarrow 3}(r_j^{\text{in}})
$$

The bounded residual is:

$$
\Delta x_j^{\text{bounded}} = \alpha \tanh(\Delta x_j)
$$

where:

$$
\alpha = \frac{3.0\ \text{mm}}{m}
$$

and $m$ is the per-sample normalization scale.

The final predicted landmark coordinate is:

$$
\hat{x}_j = \bar{x}_j + \Delta x_j^{\text{bounded}}
$$

This is the central coordinate mechanism of the method.

### Why the 3 mm bound exists

The 3 mm bound prevents the residual head from drifting too far away from the probability-supported heatmap location. It should be described as a stability constraint for bounded residual correction, not as a universal anatomical constant.

Recommended paper wording:

> The residual correction is bounded to prevent unconstrained coordinate drift away from the heatmap-supported region. The bound is used as a stability hyperparameter for local refinement around the heatmap-attention pooled coordinate.

---

## 6. Stage-Wise Auxiliary Heatmap Supervision

Current setting:

```text
use_stagewise_aux_hm = True
ablation_only = frozen_aux_drop
aux_drop_epochs = 30
```

DeepPA produces auxiliary heatmaps at internal stages through `sem_sup`. The final main heatmap is appended to the list of heatmaps.

The hierarchical heatmap loss separates:

```text
L_main_hm: final main heatmap loss
L_aux_hm : auxiliary stage heatmap loss
```

The auxiliary weight is:

$$
w_{\text{aux}}(e) = \max\left(0,\ 1 - \frac{e}{30}\right)
$$

Thus:

```text
epoch 0  : aux weight near 1.0
epoch 30 : aux weight reaches 0.0
after 30 : aux loss is not used
```

The main heatmap loss remains active.

---

## 7. Loss Function and Schedule

### Total loss

The total loss for `frozen_aux_drop` is:

$$
\mathcal{L}_{\text{total}}
= w_{\text{main}}\mathcal{L}_{\text{main-hm}}
+ w_{\text{aux}}\mathcal{L}_{\text{aux-hm}}
+ w_{\text{geom}}\mathcal{L}_{\text{geom}}
+ \mathcal{L}_{\text{frozen-PA}}
$$

For the current frozen setting:

```text
frozen_paconv_hm_weight = 0.0 by default
```

so:

$$
\mathcal{L}_{\text{frozen-PA}} = 0
$$

### Geometry loss

The geometry loss is:

$$
\mathcal{L}_{\text{geom}}
= \mathcal{L}_{\text{crd}}
+ \mathcal{L}_{\text{srf}}
+ \mathcal{L}_{\text{str}}
$$

Current modes:

```text
coord_loss_mode   = focal_l1
surface_loss_mode = topk
struct_loss_mode  = coord
```

### Coordinate loss

The coordinate loss is focal L1:

$$
\mathcal{L}_{\text{crd}}
= \frac{1}{B L}\sum_{b,j}
\left\lVert \hat{x}_{b,j} - x_{b,j}^{gt} \right\rVert_1
\left(1 + \left\lVert \hat{x}_{b,j} - x_{b,j}^{gt} \right\rVert_1\right)^\gamma
$$

with:

$$
\gamma = 2
$$

### Top-k curvature-aware surface loss

Current setting:

```text
surface_loss_mode = topk
plane_knn = 5
curv_knn = 30
curv_alpha = 10.0
dir_beta = 1.0
```

The surface loss uses hard kNN neighborhoods:

- GT local plane normal from $k_{\text{plane}} = 5$ neighbors.
- Predicted and GT curvature from $k_{\text{curv}} = 30$ neighbors.
- A point-to-plane distance weighted by curvature and direction mismatch.

Conceptually:

$$
\mathcal{L}_{\text{srf}}
= \mathbb{E}_{j}
\left[
d_{\text{p2p}}(\hat{x}_j, \Pi_j^{gt})
\left(1 + \alpha |\kappa_j^{pred} - \kappa_j^{gt}| + \beta d_{\text{dir}}\right)
\right]
$$

The actual code multiplies the point-to-plane distance by the curvature/direction multiplier:

$$
\mathcal{L}_{\text{srf}}
= \mathbb{E}_{j}
\left[
d_{\text{p2p}}
\cdot
\left(1 + \alpha \Delta\kappa + \beta \Delta d\right)
\right]
$$

where:

```text
alpha = curv_alpha = 10.0
beta  = dir_beta   = 1.0
```

### Structural loss

For `struct_loss_mode=coord`, structure is computed from pairwise landmark distances:

$$
\mathcal{L}_{\text{str}}
= \left\lVert
D(\hat{X}) - D(X^{gt})
\right\rVert_1
$$

where $D(\cdot)$ is the landmark-to-landmark pairwise distance matrix.

### Train-HM plateau schedule

Current setting:

```text
loss_schedule = train_hm_plateau
plateau_start_epoch = 30
plateau_window = 20
plateau_patience = 10
plateau_threshold = 0.01
plateau_transition_epochs = 1
plateau_transition_mode = linear
fixed_main_weight = 1.0
fixed_geom_weight = 1.0
```

Before the train main heatmap plateau trigger:

$$
w_{\text{main}} = 1.0,\quad
w_{\text{geom}} = 0.0
$$

After the plateau trigger and one-epoch transition:

$$
w_{\text{main}} = 1.0,\quad
w_{\text{geom}} = 1.0
$$

Thus, this schedule is:

1. First train with heatmap supervision only.
2. Detect when the train main heatmap loss has plateaued.
3. Keep heatmap supervision on.
4. Add coordinate/surface/structure geometry losses with weight 1.0.

Important interpretation:

```text
fixed_main_weight=1.0 does not turn off the heatmap.
fixed_geom_weight=1.0 turns on the whole geometry loss bundle:
L_crd + L_srf + L_str.
```

---

## 8. Final Training Configuration

### Current seed-controlled run command

Working directory:

```text
C:\Users\CGlab\Desktop\Earlandmark\Ear 3DFA-GCN
```

Pipeline command:

```powershell
C:\Users\CGlab\anaconda3\envs\EarLandMarking_DeepLA\python.exe run_frozen.py --need_resample False --exp_name S2G_Frozen_HMAttnResidual_SeedFixed500 --stage1_exp_name S2G_PAconv_Refinement --stage1_user_tag finetune_paconv_heatmap --train_dataset_name train --val_dataset_name valiation --test_dataset_name test --val_partition val --Eval_DataType test --epochs 500 --train_len 200 --batch_size 4 --num_points 8192 --geom_batch_size 16 --fusion_residual_base prior --ablation_only frozen_aux_drop --aux_drop_epochs 30 --use_stagewise_aux_hm True --decoder_fusion prog_half_final320 --train_coord_readout heatmap_attn_residual --hm_attn_residual_max_mm 3.0 --heatmap_activation_mode sigmoid --heatmap_loss_mode adaptive_wing --coord_loss_mode focal_l1 --struct_loss_mode coord --surface_loss_mode topk --loss_schedule train_hm_plateau --plateau_start_epoch 30 --plateau_window 20 --plateau_patience 10 --plateau_threshold 0.01 --plateau_transition_epochs 1 --plateau_transition_mode linear --fixed_main_weight 1.0 --fixed_geom_weight 1.0 --model frozen_aux_drop --user_tag pa163_hmattnres3mm_seed1_fixtrain_retry2_20260518 --latent_injection_type raw --model_epoch Single_PAConv_last.t7 --fix_train_seed True --seed 1
```

The actual `train.py` command generated by `run_frozen.py` was:

```powershell
C:\Users\CGlab\anaconda3\envs\EarLandMarking_DeepLA\python.exe train.py --exp_name S2G_Frozen_HMAttnResidual_SeedFixed500 --stage1_exp_name S2G_PAconv_Refinement --stage1_user_tag finetune_paconv_heatmap --train_dataset_name train --val_dataset_name valiation --test_dataset_name test --val_partition val --Eval_DataType test --epochs 500 --train_len 200 --batch_size 4 --num_points 8192 --geom_batch_size 16 --fusion_residual_base prior --ablation_only frozen_aux_drop --aux_drop_epochs 30 --use_stagewise_aux_hm True --decoder_fusion prog_half_final320 --train_coord_readout heatmap_attn_residual --hm_attn_residual_max_mm 3.0 --heatmap_activation_mode sigmoid --heatmap_loss_mode adaptive_wing --coord_loss_mode focal_l1 --struct_loss_mode coord --surface_loss_mode topk --loss_schedule train_hm_plateau --plateau_start_epoch 30 --plateau_window 20 --plateau_patience 10 --plateau_threshold 0.01 --plateau_transition_epochs 1 --plateau_transition_mode linear --fixed_main_weight 1.0 --fixed_geom_weight 1.0 --fix_train_seed True --seed 1 --need_resample False --model frozen_aux_drop --user_tag pa163_hmattnres3mm_seed1_fixtrain_retry2_20260518_Stage2_RAW_AuxDrop30 --latent_injection_type raw --model_epoch Single_PAConv_last.t7
```

### Seed control

The seed-fixed run used:

```text
--fix_train_seed True
--seed 1
```

This sets:

```text
PYTHONHASHSEED
Python random seed
NumPy seed
torch.manual_seed
torch.cuda.manual_seed
torch.cuda.manual_seed_all
cudnn.benchmark = False
cudnn.deterministic = True
```

This is a seed-controlled run, but not guaranteed bit-level deterministic because custom CUDA extensions and some GPU operations may remain nondeterministic.

### Extension loading note

This run used an existing prebuilt `cutils_.pyd` to avoid rebuilding the custom CUDA extension in the current Windows/MSVC environment:

```text
DEEPLA_CUTILS_PREBUILT_DIR=C:\Users\CGlab\Desktop\Earlandmark\Ear 3DFA-GCN\debug_outputs\torch_ext_pa163_hmattnres3mm\cutils_
```

This is infrastructure-level extension loading. It does not change:

- model architecture,
- loss functions,
- metric calculations,
- dataset split,
- checkpoint contents.

---

## 9. Latest Seed-Fixed Run Result

### Output folder

```text
C:\Users\CGlab\Desktop\Earlandmark\results\S2G_Frozen_HMAttnResidual_SeedFixed500\FPS8192_sigma2.5_batch4_train200_pa163_hmattnres3mm_seed1_fixtrain_retry2_20260518_Stage2_RAW_AuxDrop30_1
```

### HMR checkpoint

```text
models\Frozen_Aux_Drop_last.t7
```

SHA256:

```text
FC50E3966AF346D09ECD3197008600BBFCC293C39DB317B4C41B4F00A09D4432
```

### Evaluation result

Result files:

```text
frozen_aux_drop_Results_ME1.4737.xlsx
frozen_aux_drop_Results_ME1.4737.txt
```

Summary:

```text
Average ME             : 1.4737 +/- 0.8780 mm
95%ile ME              : 1.8958 mm
Success Rate (<10mm)   : 100.00 %
Success Rate (<5mm)    : 100.00 %
Heatmap Cosine Sim     : 69.04 %
Heatmap mIoU (@0.1)    : 25.00 %
Average Inference Time : 107.06 ms
```

Surface diagnostics:

```text
Surface Loss Mode       : topk
Surface Loss (norm)     : 0.008394
Surface Normal Distance : 0.2245 mm
Surface Curvature Diff  : 0.008751
Surface Direction Diff  : 0.280528
```

### Predicted landmark ASC files

The evaluation generated one `.asc` prediction file per test sample:

```text
LM\frozen_aux_drop\p_*.asc
```

Count:

```text
43 predicted landmark ASC files
```

This matches the test split size:

```text
test samples = 43
```

Each `.asc` contains the denormalized predicted 3D landmark coordinates for one ear sample:

```text
x, y, z
...
36 rows total
```

---

## 10. Evaluation Flow

Evaluation is run by `eval.py` after training completes through `run_frozen.py`.

The evaluation procedure is:

1. Load `npy_data` backup from the run folder.
2. Load test point cloud, GT landmarks, GT heatmaps, and sample names.
3. Normalize input point cloud and GT landmarks.
4. Run Stage1 PAConv and Stage2 DeepPA.
5. Get normalized predicted coordinates.
6. Denormalize predictions:

$$
\hat{X}_{mm} = \hat{X}_{norm} \cdot m + c
$$

where:

- $m$ is the sample normalization radius.
- $c$ is the sample centroid.

7. Compute per-landmark Euclidean distance:

$$
e_j = \left\lVert \hat{x}_j - x_j^{gt} \right\rVert_2
$$

8. Compute:

$$
\text{ME} = \frac{1}{L}\sum_{j=1}^{L} e_j
$$

9. Save:

- result `.xlsx`,
- result `.txt`,
- heatmap visualization images,
- predicted landmark `.asc` files.

Important:

```text
Training-time Val_mm does not save ASC files.
Formal eval.py saves ASC files.
```

---

## 11. How To Explain The Architecture In A Paper

### Short method description

> We use a frozen PAConv model as a dense geometric prior and inject its latent representation into a hierarchical DeepPA refinement network. The refined decoder feature predicts a dense landmark heatmap. Final landmark coordinates are obtained by heatmap-attention pooling over point coordinates and features, followed by a bounded residual correction. This design keeps coordinate regression anchored to the heatmap-supported point cloud region while allowing local geometric refinement.

### More detailed method description

> Given a 7-channel point cloud input containing XYZ coordinates, principal local direction, and curvature, a frozen PAConv model first extracts a 1344-channel dense prior. The prior is projected into each DeepPA stage and fused through a prior-centered residual interaction module. The Stage2 decoder outputs a 320-channel dense feature, which is concatenated with XYZ coordinates and passed to a heatmap head. The predicted landmark heatmap is used as an attention distribution over points. For each landmark, we pool both the 3D coordinates and the 512-dimensional head feature using the heatmap attention weights. A residual MLP predicts a local coordinate correction from the concatenated 515-dimensional pooled descriptor. The correction is bounded by a 3 mm tanh clamp and added to the pooled coordinate.

### Recommended phrasing for the 3 mm residual bound

Do not describe 3 mm as an anatomical law. Describe it as a local stability bound:

> The residual correction is bounded to encourage local refinement around the heatmap-supported estimate and to prevent unconstrained coordinate drift.

### Recommended phrasing for reproducibility

> All downstream comparisons are performed using the same fixed PAConv prior checkpoint. We provide the trained HMR checkpoint for checkpoint-level reproduction. From-scratch Stage2 training shows mild stochastic variation, especially because geometry loss activation is triggered by the training heatmap plateau.

---

## 12. Diagram Checklist

For the main architecture figure:

- Show input as `XYZ + direction + curvature`, not GT landmarks.
- Do not draw GT landmarks as model input.
- If GT landmarks are shown, put them only in a separate training-supervision/loss inset.
- Show PAConv prior as `1344 x N`.
- Show prior projection into stage dimensions.
- Show DeepPA hierarchy:

```text
8192 -> 2048 -> 512 -> 128 -> 32
```

- Show decoder final feature as `320 x N`.
- Show head input as:

```text
320 feature + 3 XYZ = 323 x N
```

- Show heatmap logits/probability:

```text
36 x N
```

- Show attention pooling:

```text
pooled feature: 512
pooled XYZ: 3
concat: 515
```

- Show residual MLP:

```text
515 -> 256 -> 3
```

- Show final coordinate:

$$
\hat{x}_j = \bar{x}_j + 3\text{mm}\cdot\tanh(\Delta x_j)
$$

---

## 13. Comparison Numbers To Keep Straight

### Best saved checkpoint

```text
Average ME = 1.4317 +/- 0.8755 mm
95%ile ME = 1.8638 mm
```

This is the best saved HMR checkpoint evaluation and should be used carefully as the best reported checkpoint result.

### Latest seed-controlled retraining

```text
Average ME = 1.4737 +/- 0.8780 mm
95%ile ME = 1.8958 mm
```

This is the current seed=1 Stage2 from-scratch retraining result under the same main setting.

### Interpretation

The method remains better than plain PAConv and competitive with previous HMR results, but from-scratch Stage2 retraining variance should be acknowledged. For paper positioning, emphasize fixed-prior checkpoint reproduction and ablation trends rather than claiming that every from-scratch run exactly returns 1.4317.

---

## 14. Files To Cite Internally

Model and training:

```text
train.py
DeepPA_model.py
deeppa_semseg.py
PAConv_model.py
loss.py
loss_controller.py
augmentations.py
dataset.py
util.py
run_frozen.py
eval.py
```

Latest seed-fixed result:

```text
C:\Users\CGlab\Desktop\Earlandmark\results\S2G_Frozen_HMAttnResidual_SeedFixed500\FPS8192_sigma2.5_batch4_train200_pa163_hmattnres3mm_seed1_fixtrain_retry2_20260518_Stage2_RAW_AuxDrop30_1
```

Latest predicted landmark files:

```text
C:\Users\CGlab\Desktop\Earlandmark\results\S2G_Frozen_HMAttnResidual_SeedFixed500\FPS8192_sigma2.5_batch4_train200_pa163_hmattnres3mm_seed1_fixtrain_retry2_20260518_Stage2_RAW_AuxDrop30_1\LM\frozen_aux_drop\p_*.asc
```

---

## 15. Practical Run Notes

### Do not resample for reproduction

Use:

```text
--need_resample False
```

This ensures the run uses the existing cached `npy_data`.

### Do not train PAConv for the final HMR comparison

The final HMR comparison should use the fixed PAConv prior:

```text
--stage1_exp_name S2G_PAconv_Refinement
--stage1_user_tag finetune_paconv_heatmap
--model_epoch Single_PAConv_last.t7
```

### Do not run eval concurrently during training

During training, `train.py` computes `Val_mm` internally. This is not formal `eval.py`, and it does not save `.asc` files. Wait for training completion, then `run_frozen.py` launches eval and creates:

```text
Results_ME*.txt
Results_ME*.xlsx
LM/*.asc
```

### Verify an evaluation run

A complete evaluation should have:

```text
1 txt result file
1 xlsx result file
43 predicted ASC files for the test split
```

For the latest seed-fixed run, this condition is satisfied.
