# PEARL Final

This directory records the configuration used by the reported PEARL result:

- Test ME: 1.3948 mm
- Test STD: 1.2326 mm
- Test p95: 3.9724 mm
- Surface ME: 0.2073 mm

## Included

- `pearl_final_13948.json`: machine-readable final parameters.
- `run_pearl_final.ps1`: training and best-checkpoint evaluation entry point.

The dataset and model checkpoints are intentionally not included in Git.

## Required Stage-1 Prior

Use the paper Stage-1 checkpoint:

```text
Original_PAConv_best.t7
SHA-256: B7B2B7FFB3EFDD4C6CAC0F33478BAD86E0D01B97D99000D3572DB034B1C969E2
```

Verify it on Windows:

```powershell
Get-FileHash -Algorithm SHA256 .\Original_PAConv_best.t7
```

## Dataset Layout

The existing loader expects these dataset names under the configured data root:

```text
train      200 scans
valiation   53 scans
test        43 scans
```

`valiation` is the existing directory name and is preserved for compatibility.
Each sample uses 8,192 FPS points and 36 landmarks.

## Train

Activate the project Conda environment and Visual Studio 2019 developer
environment first. From the repository root:

```powershell
.\release\PEARL_FINAL_SUBCOM1\run_pearl_final.ps1 `
  -Stage1CheckpointPath "C:\path\to\Original_PAConv_best.t7" `
  -DataRoot "C:\path\to\data" `
  -OutputRoot ".\results\PEARL_FINAL"
```

## Evaluate Best Checkpoint

After training:

```powershell
.\release\PEARL_FINAL_SUBCOM1\run_pearl_final.ps1 `
  -Stage1CheckpointPath "C:\path\to\Original_PAConv_best.t7" `
  -DataRoot "C:\path\to\data" `
  -OutputRoot ".\results\PEARL_FINAL" `
  -EvalBest
```

## Final Configuration Summary

- Training seed: 6
- Dataset seed: 1
- Epochs: 500
- Batch size: 4
- Input points: 8,192
- Heatmap sigma: 2.5
- Stage-1 PAConv input: XYZ, 3 channels
- Stage-2 PEARL input: 7 channels
- Stage-1 prior: frozen
- Decoder: dense sigmoid heatmap attention plus feature-conditioned residual
- Aux heatmap: linear drop for 30 epochs
- Plateau schedule: 20-epoch window, patience 10, threshold 0.01
- Coordinate, surface, and structure term weights: 1.0 each
- Surface plane neighborhood: 5
- Curvature neighborhood: 30
- Curvature alpha: 10
- Direction beta: 1

The reported result is tied to this exact split, prior checkpoint, and seed.
