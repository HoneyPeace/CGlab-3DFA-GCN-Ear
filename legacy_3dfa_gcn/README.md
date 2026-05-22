# Original 3DFA-GCN PAConv Reproduction

This folder keeps a local reproduction path for the original 3DFA-GCN PAConv
heatmap model while reading the current Earlandmark `train/val/test` NPY splits.

The reproduction script intentionally stays separate from the main DeepPA/PAConv
pipeline so existing experiments, metrics, and model code are not changed.

Key differences from the current PAConv path:

- Input point features are XYZ only.
- The first edge convolution uses the original 6-channel edge feature:
  `neighbor_xyz - center_xyz` plus `center_xyz`.
- ScoreNet receives the original 10-channel geometric score feature.
- The PAConv head returns raw heatmap logits, matching the original repository.
- Training uses heatmap loss only.
- Paper-aligned defaults use `k=30` graph neighbors and `r=10` regression
  points for local surface unfolding.
- MDS evaluation defaults to the paper Eq. 8 distance weighting. Use
  `--mds_distance_weight none` for the released GitHub code path without
  weighted MDS.
- Final evaluation reports both top-k weighted pooling and MDS landmark
  regression from the same predicted heatmap.

Paper-aligned command for the current split:

```powershell
python legacy_3dfa_gcn\original_paconv_current_split.py `
  --use_cuda_extension `
  --val_method mds `
  --mds_distance_weight paper `
  --k 30 `
  --regression_point_num 10 `
  --epochs 500 `
  --batch_size 4 `
  --seed 1 `
  --loss adaptive_wing
```
