# PA163 Softmax Heatmap Loss Branch

Date: 2026-05-13

## Purpose

This branch prepares the next experiment after the HMAttnResidual AuxDrop500
result (`ME 1.4596 mm`, `STD 0.8834 mm`).

The new options allow main and auxiliary DeepPA heatmap outputs to use a
point-wise softmax activation while keeping Adaptive Wing loss as the heatmap
criterion. A separate `softmax_ce` heatmap loss mode remains available, but it
is not the target setting for the immediate comparison queue.

## New Flags

- `--heatmap_activation_mode softmax`
- `--heatmap_activation_temperature 1.0`
- `--heatmap_loss_mode adaptive_wing`

## Preserved Behavior

- Default heatmap activation remains `sigmoid`.
- Default heatmap supervision remains `adaptive_wing`.
- Existing evaluation behavior is preserved by default; with
  `--heatmap_activation_mode softmax`, the returned main/aux heatmaps follow the
  same softmax activation used during training.
- Coordinate readout, surface loss, structural loss, metrics, and result formats
  are unchanged unless separately configured.
