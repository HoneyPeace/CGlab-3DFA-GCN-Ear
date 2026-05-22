$ErrorActionPreference = "Stop"

# Main-computer DeepPA follow-up queue after the 14ch PAConv/DeepPA baseline decision.
# This queue keeps both PAConv and DeepPA in center_geometry mode:
# - raw NPY input: 7ch = XYZ + principal direction 3ch + curvature 1ch
# - internal local feature: 14ch = XYZ relation + center principal direction/curvature
# The first run trains/reuses one PAConv Stage1 tag, and later runs reuse that Stage1.

$Runner = Join-Path $PSScriptRoot "run_paconv_deeppa_featuremode_seed1.ps1"
$Stage1UserTag = "main14_center_stage1_seed1"

$Runs = @(
    @{
        Name = "base_auxdrop_plateau_fps_hmr2_seed1"
        AblationOnly = "frozen_aux_drop"
        StageDownsampleMethod = "fps"
        LossSchedule = "train_hm_plateau"
        HmResidualMm = 2.0
    },
    @{
        Name = "auxfixed_plateau_fps_hmr2_seed1"
        AblationOnly = "frozen_aux_fixed"
        StageDownsampleMethod = "fps"
        LossSchedule = "train_hm_plateau"
        HmResidualMm = 2.0
    },
    @{
        Name = "noaux_plateau_fps_hmr2_seed1"
        AblationOnly = "frozen_no_aux"
        StageDownsampleMethod = "fps"
        LossSchedule = "train_hm_plateau"
        HmResidualMm = 2.0
    },
    @{
        Name = "auxdrop_fixed60_fps_hmr2_seed1"
        AblationOnly = "frozen_aux_drop"
        StageDownsampleMethod = "fps"
        LossSchedule = "fixed_three_phase"
        HmResidualMm = 2.0
    },
    @{
        Name = "auxdrop_plateau_grid_hmr2_seed1"
        AblationOnly = "frozen_aux_drop"
        StageDownsampleMethod = "grid"
        LossSchedule = "train_hm_plateau"
        HmResidualMm = 2.0
    },
    @{
        Name = "auxdrop_plateau_fps_hmr1p5_seed1"
        AblationOnly = "frozen_aux_drop"
        StageDownsampleMethod = "fps"
        LossSchedule = "train_hm_plateau"
        HmResidualMm = 1.5
    }
)

Write-Output "[QUEUE] Main14 DeepPA follow-up queue"
Write-Output "[QUEUE] Stage1 tag reused by all variants: $Stage1UserTag"
Write-Output "[QUEUE] Variant count: $($Runs.Count)"

foreach ($Run in $Runs) {
    Write-Output "[QUEUE] Starting variant: $($Run.Name)"
    & $Runner `
        -PaconvFeatureMode center_geometry `
        -DeepPAFeatureMode center_geometry `
        -RunSuffix $Run.Name `
        -Stage1UserTag $Stage1UserTag `
        -AblationOnly $Run.AblationOnly `
        -StageDownsampleMethod $Run.StageDownsampleMethod `
        -LossSchedule $Run.LossSchedule `
        -HmResidualMm $Run.HmResidualMm `
        -Seed 1 `
        -DatasetSeed 1
    Write-Output "[QUEUE] Finished variant: $($Run.Name)"
}

Write-Output "[QUEUE_DONE] Main14 DeepPA follow-up queue completed"
