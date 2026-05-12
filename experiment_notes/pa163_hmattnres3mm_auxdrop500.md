# PA163 HMAttnResidual 3mm AuxDrop500 Result

Date: 2026-05-13

## Experiment

- Exp name: `S2G_Frozen_HMAttnResidual_AuxDrop500`
- User tag: `pa163_hmattnres3mm_auxdrop30_trainhmplateau_topkcurv_500`
- Model: Frozen PAConv prior + DeepPA `prog_half_final320`
- Coordinate readout: `heatmap_attn_residual`
- Residual limit: `3.0 mm`
- Aux schedule: `frozen_aux_drop`, `aux_drop_epochs=30`
- Loss schedule: `train_hm_plateau`
- Surface loss: `topk`
- Epochs: `500`
- Batch size: `4`

## Final Evaluation

- ME: `1.4596 mm`
- STD: `0.8834 mm`
- 95%ile: `2.0678 mm`

## Reference

Earlier HMAttnResidual 100-epoch aux-fixed run:

- ME: `1.7548 mm`
- STD: `0.9700 mm`

Improvement:

- ME improvement: `0.2952 mm`

## Note

The final score narrowly missed the current target of `1.45 mm` by `0.0096 mm`.
This result motivates the next branch experiment: replacing sigmoid/Adaptive-Wing
heatmap supervision with point-wise softmax distribution supervision for both
main and auxiliary heatmaps.
