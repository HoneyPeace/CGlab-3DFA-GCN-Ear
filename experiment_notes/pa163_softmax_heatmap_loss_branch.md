# PA163 Softmax Heatmap Loss Branch

Date: 2026-05-13

## Purpose

This branch prepares the next experiment after the HMAttnResidual AuxDrop500
result (`ME 1.4596 mm`, `STD 0.8834 mm`).

The new option changes main and auxiliary DeepPA heatmap supervision from
sigmoid + Adaptive Wing loss to a point-wise softmax distribution loss over the
full point set.

## New Flags

- `--heatmap_loss_mode softmax_ce`
- `--heatmap_softmax_temperature 1.0`

## Preserved Behavior

- Default heatmap supervision remains `adaptive_wing`.
- Evaluation heatmaps still use the existing sigmoid output path.
- Coordinate readout, surface loss, structural loss, metrics, and result formats
  are unchanged unless separately configured.
