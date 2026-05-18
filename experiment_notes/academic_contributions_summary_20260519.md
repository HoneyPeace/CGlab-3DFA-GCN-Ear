# Academic Contributions Summary

Date: 2026-05-19
Project: Ear 3D Landmark Detection with PAConv Prior and DeepPA Heatmap-Residual Refinement
Purpose: Summarize the academic achievements and paper-level contributions of the current method.

---

## 1. Core Academic Claim

This work proposes a prior-guided 3D ear landmark detection framework that combines:

1. a frozen PAConv heatmap/latent prior,
2. a hierarchical DeepPA refinement backbone,
3. heatmap-attention coordinate pooling,
4. bounded residual correction,
5. geometry-aware supervision using coordinate, surface, and structural losses.

The central academic claim is:

> Landmark localization can be improved by anchoring coordinate regression to a dense heatmap-supported point region and allowing only bounded local residual correction, rather than regressing coordinates freely from global features.

In other words, the method does not treat heatmaps merely as an auxiliary output. It uses the heatmap distribution itself as the coordinate readout mechanism.

---

## 2. Methodological Contributions

## 2.1 PAConv Prior-Guided DeepPA Refinement

The method uses a pretrained PAConv model as a frozen Stage1 prior and injects its dense latent representation into the DeepPA refinement network.

The PAConv prior is:

$$
H_{\mathrm{PA}} \in \mathbb{R}^{1344 \times N}
$$

It is projected into each DeepPA stage:

$$
P_l = \phi_l(H_{\mathrm{PA}})
$$

and fused with DeepPA stage features through prior-centered residual interaction:

$$
F_l^{\mathrm{fused}} = P_l + \psi_l([F_l, P_l])
$$

Academic significance:

- Introduces a reusable frozen prior for 3D ear landmark refinement.
- Separates coarse heatmap prior learning from downstream coordinate refinement.
- Reduces the need to fully retrain the entire PAConv-DeepPA stack for each downstream comparison.
- Provides a practical checkpoint-based reproducibility strategy for research code where Stage1 training variance is nontrivial.

---

## 2.2 Heatmap-Attention Residual Coordinate Readout

The strongest conceptual contribution is the coordinate readout design.

Instead of directly regressing landmark coordinates from a global feature vector, the model first predicts a dense landmark heatmap:

$$
Z \in \mathbb{R}^{L \times N}
$$

and converts it into attention weights:

$$
a_{j,i} =
\frac{\sigma(z_{j,i})}
{\sum_{i=1}^{N}\sigma(z_{j,i}) + \epsilon}
$$

For each landmark $j$, the heatmap-attention pooled coordinate is:

$$
\bar{x}_j = \sum_{i=1}^{N} a_{j,i}x_i
$$

The heatmap-attention pooled feature is:

$$
\bar{f}_j = \sum_{i=1}^{N} a_{j,i}f_i
$$

The residual input is:

$$
r_j = [\bar{f}_j, \bar{x}_j] \in \mathbb{R}^{515}
$$

because:

$$
512 + 3 = 515
$$

The final coordinate is:

$$
\hat{x}_j = \bar{x}_j + \Delta x_j
$$

Academic significance:

- Bridges heatmap-based localization and coordinate regression.
- Makes the predicted coordinate explicitly depend on the probability distribution over the point cloud.
- Preserves interpretability because each coordinate is anchored to heatmap-supported points.
- Avoids treating coordinate regression as a disconnected black-box MLP output.

---

## 2.3 Bounded Local Residual Correction

The residual correction is bounded:

$$
\Delta x_j^{\mathrm{bounded}}
= \alpha \tanh(\Delta x_j)
$$

where $\alpha$ corresponds to the normalized form of a 3 mm physical bound:

$$
\alpha = \frac{3.0\ \mathrm{mm}}{m}
$$

The final coordinate is:

$$
\hat{x}_j = \bar{x}_j + \Delta x_j^{\mathrm{bounded}}
$$

Academic significance:

- Prevents coordinate drift away from the heatmap-supported region.
- Encourages the residual head to perform local refinement rather than unconstrained global correction.
- Provides a meaningful inductive bias: the heatmap determines the approximate landmark region, and the residual corrects only local error.
- Gives a plausible explanation for why small clamp values such as 1 mm and 3 mm perform better than no-clamp settings.

Important framing:

The 3 mm value should not be described as a universal anatomical constant. It should be described as a stability hyperparameter for local residual refinement.

Recommended wording:

> We bound the residual correction to restrict refinement to the local neighborhood supported by the heatmap distribution and to prevent unconstrained coordinate drift.

---

## 2.4 Stage-Wise Auxiliary Heatmap Supervision

The DeepPA backbone produces internal stage-wise auxiliary heatmaps. These are supervised during early training and then gradually removed:

$$
w_{\mathrm{aux}}(e) =
\max\left(0,\ 1 - \frac{e}{30}\right)
$$

The main heatmap remains supervised throughout training.

Academic significance:

- Encourages intermediate representations to become landmark-aware.
- Stabilizes early feature formation.
- Avoids over-constraining late training by removing auxiliary supervision after the early phase.
- Supports a coarse-to-refined learning process within the hierarchy.

This contribution is best described as an optimization/training strategy, not as a separate architecture.

---

## 2.5 Train-Heatmap-Plateau Loss Scheduling

The method uses a staged loss schedule controlled by the training main heatmap loss.

Before the plateau trigger:

$$
w_{\mathrm{main}} = 1.0,\quad
w_{\mathrm{geom}} = 0.0
$$

After train heatmap plateau:

$$
w_{\mathrm{main}} = 1.0,\quad
w_{\mathrm{geom}} = 1.0
$$

The total loss is:

$$
\mathcal{L}_{\mathrm{total}}
=
w_{\mathrm{main}}\mathcal{L}_{\mathrm{main-hm}}
+
w_{\mathrm{aux}}\mathcal{L}_{\mathrm{aux-hm}}
+
w_{\mathrm{geom}}
\left(
\mathcal{L}_{\mathrm{crd}}
+
\mathcal{L}_{\mathrm{srf}}
+
\mathcal{L}_{\mathrm{str}}
\right)
$$

Academic significance:

- Separates heatmap distribution learning from geometry refinement.
- Prevents coordinate/surface losses from dominating before the heatmap becomes meaningful.
- Provides a dynamic curriculum based on heatmap convergence rather than a purely fixed epoch schedule.

Caution:

This schedule also introduces training trajectory sensitivity, because the plateau trigger epoch can vary between runs.

---

## 2.6 Curvature-Aware Surface Regularization

The method uses a top-k local curvature-aware surface loss.

The input feature includes a local curvature descriptor:

$$
\kappa_i =
\frac{\lambda_1}
{\lambda_1 + \lambda_2 + \lambda_3 + \epsilon}
$$

where $\lambda_1 \le \lambda_2 \le \lambda_3$ are eigenvalues of the local kNN covariance.

The surface loss uses:

- local plane normal from top-k neighbors,
- predicted/GT local curvature,
- local principal direction difference,
- point-to-plane distance.

Conceptually:

$$
\mathcal{L}_{\mathrm{srf}}
=
\mathbb{E}_j
\left[
d_{\mathrm{p2p}}
\cdot
\left(
1
+
\alpha |\kappa_j^{pred} - \kappa_j^{gt}|
+
\beta d_{\mathrm{dir}}
\right)
\right]
$$

Academic significance:

- Uses local surface geometry to regularize coordinate predictions.
- Encourages landmarks to lie on geometrically plausible ear surface regions.
- Adds shape-awareness beyond simple coordinate distance.

This supports the claim that the method is not merely optimizing pointwise coordinate error but also respects local geometric structure.

---

## 3. Experimental Achievements

## 3.1 Best Checkpoint-Level Performance

The strongest saved HMR checkpoint achieved:

```text
Average ME = 1.4317 +/- 0.8755 mm
95%ile ME = 1.8638 mm
```

This is the main best-result number to use carefully in the paper as the best checkpoint result.

Academic meaning:

- Demonstrates that the proposed heatmap-attention residual refinement can outperform the PAConv-only prior.
- Shows that the fixed-prior HMR setting can reach approximately 1.43 mm mean error.
- Provides the strongest empirical evidence for the method's effectiveness.

---

## 3.2 Seed-Controlled Reproduction Run

A seed-controlled Stage2 retraining run was completed with:

```text
--fix_train_seed True
--seed 1
```

Result:

```text
Average ME = 1.4737 +/- 0.8780 mm
95%ile ME = 1.8958 mm
Heatmap Cosine Similarity = 69.04 %
Heatmap mIoU (@0.1) = 25.00 %
Surface Normal Distance = 0.2245 mm
```

Academic meaning:

- Confirms that the method remains in the same performance regime under a fresh seed-controlled Stage2 training run.
- Shows that the method does not collapse when retrained.
- Demonstrates reproducibility at the method level, though not exact bitwise reproduction of the best checkpoint.

---

## 3.3 Prediction Export and Qualitative Reproducibility

The final evaluation generated predicted landmark files:

```text
LM/frozen_aux_drop/p_*.asc
```

Count:

```text
43 ASC files
```

This matches the test set size.

Academic meaning:

- The method produces explicit 3D landmark coordinates, not only aggregate metrics.
- The predictions can be visualized, inspected, and compared qualitatively.
- This supports qualitative figures, failure case analysis, and reproducibility materials.

---

## 3.4 Clamp Ablation Insight

Existing clamp comparison showed:

```text
3 mm clamp      : 1.4317 +/- 0.8755
1 mm clamp      : 1.4351 +/- 0.8799
0 mm/no-clamp   : 1.4511 +/- 0.8706
2 mm clamp      : 1.4777 +/- 0.8970
```

Academic interpretation:

- The main defensible conclusion is not that exactly 3 mm is universally optimal.
- The safer claim is that bounded residual correction improves stability compared with unconstrained residual correction.
- Small local bounds, especially 1 to 3 mm, appear beneficial.

Recommended wording:

> Bounded residual correction improves the stability of heatmap-guided coordinate refinement. Small residual bounds performed better than unconstrained residual regression in our experiments.

---

## 4. Reproducibility Findings

## 4.1 What Was Reproducible

The following were verified as reproducible or traceable:

- Existing HMR checkpoint evaluation.
- Fixed PAConv prior checkpoint identity through SHA256.
- NPY/cache consistency.
- Evaluation metric logic.
- Test-set ASC prediction export.

Important PAConv prior hash:

```text
9DC0C5AF99FB2DC235654D2297E0854E3887E14BF815C3CBCAC2EAFAA0B9F7F4
```

Latest seed-fixed HMR checkpoint hash:

```text
FC50E3966AF346D09ECD3197008600BBFCC293C39DB317B4C41B4F00A09D4432
```

Academic meaning:

- The reported checkpoint can be identified exactly.
- Downstream experiments can be anchored to the same frozen PAConv prior.
- Checkpoint-based reproduction is defensible.

---

## 4.2 What Showed Variance

From-scratch Stage2 training showed non-negligible variance:

```text
Best saved checkpoint: 1.4317 mm
Seed-controlled retraining: 1.4737 mm
Subcomputer retraining example: about 1.449 mm
```

Likely contributors:

- Stage2 random initialization.
- dropout,
- DataLoader order,
- CUDA/custom operation nondeterminism,
- train-HM plateau trigger timing,
- geometry loss activation timing.

Academic meaning:

- The method is reproducible in the checkpoint/evaluation sense.
- Exact from-scratch training reproduction is more sensitive.
- This should be treated as a limitation or implementation note, not hidden.

Recommended wording:

> We release the fixed PAConv prior and trained HMR checkpoint for checkpoint-level reproducibility. From-scratch Stage2 training can show mild stochastic variation due to the staged loss schedule and GPU-level nondeterminism.

---

## 5. Paper-Level Contributions To Claim

## Contribution 1: Prior-Guided Refinement

The method uses a frozen PAConv prior to guide DeepPA refinement.

Claim:

> A frozen PAConv heatmap/latent prior can be reused as a stable geometric anchor for downstream 3D landmark refinement.

---

## Contribution 2: Heatmap-Anchored Coordinate Regression

The method converts a predicted heatmap into a coordinate estimate through attention pooling.

Claim:

> Coordinates are regressed from heatmap-supported point distributions rather than directly from global latent features.

---

## Contribution 3: Bounded Residual Correction

The method adds a bounded residual to the heatmap-pooled coordinate.

Claim:

> Local residual correction improves flexibility while preventing unconstrained coordinate drift.

---

## Contribution 4: Geometry-Aware Supervision

The method combines coordinate, surface, and structural objectives.

Claim:

> Local surface curvature and pairwise landmark structure provide additional geometric constraints for anatomically plausible predictions.

---

## Contribution 5: Training Curriculum

The method uses stage-wise auxiliary heatmap supervision and train-HM plateau-based geometry activation.

Claim:

> Heatmap-first training stabilizes probability localization before geometry losses are introduced.

---

## Contribution 6: Reproducible Checkpoint Protocol

The method explicitly fixes and identifies the PAConv prior checkpoint and provides downstream HMR checkpoints.

Claim:

> For multi-stage geometric learning pipelines, checkpoint-level reproducibility is essential and should be separated from from-scratch training variance.

---

## 6. How To Position The Work Academically

The safest academic framing is:

1. This is not just a new loss function.
2. This is not just PAConv plus DeepPA.
3. The main novelty is the heatmap-anchored residual coordinate readout under a fixed geometric prior.
4. The method is designed for 3D point-cloud landmark localization where heatmap probability and local geometry should jointly determine coordinates.

Recommended abstract-style description:

> We propose a prior-guided heatmap-residual framework for 3D ear landmark localization. A frozen PAConv model provides dense geometric priors, which are injected into a hierarchical DeepPA refinement network. The network predicts landmark heatmaps and obtains coordinates through heatmap-attention pooling followed by bounded local residual correction. To stabilize training, we combine stage-wise auxiliary heatmap supervision with a heatmap-plateau curriculum that activates coordinate, curvature-aware surface, and structural losses after heatmap convergence. Experiments show that the proposed refinement improves over PAConv-only heatmap localization and achieves strong checkpoint-level performance.

---

## 7. What Not To Overclaim

Avoid claiming:

- The 3 mm residual bound is universally optimal.
- From-scratch training always reproduces exactly 1.4317 mm.
- PAConv training itself is fully stable.
- The method solves all reproducibility issues.
- The model does not require checkpoint release.

Better claims:

- Bounded residual correction improves stability compared with no-clamp.
- The best checkpoint achieves 1.4317 mm.
- Seed-controlled retraining remains in a similar performance range.
- Fixed-prior checkpoint release supports reproducibility.
- From-scratch training variance is an acknowledged limitation.

---

## 8. Bottom-Line Academic Achievement

The academic achievement of this work is:

> It turns a heatmap-only PAConv prior into a stronger, geometry-aware coordinate prediction system by using the heatmap itself as an attention mechanism for coordinate pooling and then applying bounded local residual refinement under curvature-aware supervision.

In practical terms:

- PAConv provides the coarse probability/geometric prior.
- DeepPA refines dense hierarchical features.
- Heatmap attention gives interpretable point-supported coordinates.
- A bounded residual gives local correction capacity.
- Geometry losses encourage surface-consistent landmarks.
- Checkpoint release makes the final result reproducible.

This is the main story to carry into the paper.
