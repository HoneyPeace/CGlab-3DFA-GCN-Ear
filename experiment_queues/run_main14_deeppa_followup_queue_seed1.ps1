$ErrorActionPreference = "Stop"

# Main-computer DeepPA follow-up queue after the 14ch PAConv/DeepPA baseline decision.
# This queue keeps both PAConv and DeepPA in center_geometry mode:
# - raw NPY input: 7ch = XYZ + principal direction 3ch + curvature 1ch
# - internal local feature: 14ch = XYZ relation + center principal direction/curvature
# It loads the PAConv-only command's latest Single_PAConv_last.t7 as Stage1.

$Runner = Join-Path $PSScriptRoot "run_paconv_deeppa_featuremode_seed1.ps1"
$Stage1UserTag = "main14_center_stage1_seed1"
$RepoDir = Split-Path -Parent $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $RepoDir
$OutputRootBase = Join-Path $WorkspaceRoot "results\MainPAConv_DeepPAReady"
$PaconvExpName = "S2G_MainPAConv_Raw7_CenterGeom_MDS_Seed1"
$PaconvRunTag = "mainpaconv_raw7_centergeom_mds_seed1"
$PaconvExpDir = Join-Path $OutputRootBase $PaconvExpName

function Find-LatestPaconvLastCheckpoint {
    param(
        [string]$ExperimentDir,
        [string]$RunTag
    )

    if (-not (Test-Path -LiteralPath $ExperimentDir)) {
        throw "PAConv result folder not found. Run .\experiment_queues\run_mainpaconv_raw7_centergeom_seed1.ps1 first. Missing: $ExperimentDir"
    }

    $runDirs = Get-ChildItem -LiteralPath $ExperimentDir -Directory |
        Where-Object {
            $_.Name -like "*$RunTag*" -and
            (Test-Path -LiteralPath (Join-Path $_.FullName "models\Single_PAConv_last.t7"))
        } |
        Sort-Object LastWriteTime -Descending

    if (-not $runDirs -or $runDirs.Count -eq 0) {
        throw "Single_PAConv_last.t7 not found for $RunTag. Run the PAConv-only command/eval first."
    }

    return Join-Path $runDirs[0].FullName "models\Single_PAConv_last.t7"
}

$Stage1CheckpointPath = Find-LatestPaconvLastCheckpoint -ExperimentDir $PaconvExpDir -RunTag $PaconvRunTag

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
Write-Output "[QUEUE] Stage1 checkpoint loaded by all variants: $Stage1CheckpointPath"
Write-Output "[QUEUE] Variant count: $($Runs.Count)"

foreach ($Run in $Runs) {
    Write-Output "[QUEUE] Starting variant: $($Run.Name)"
    & $Runner `
        -PaconvFeatureMode center_geometry `
        -DeepPAFeatureMode center_geometry `
        -RunSuffix $Run.Name `
        -Stage1UserTag $Stage1UserTag `
        -Stage1CheckpointPath $Stage1CheckpointPath `
        -AblationOnly $Run.AblationOnly `
        -StageDownsampleMethod $Run.StageDownsampleMethod `
        -LossSchedule $Run.LossSchedule `
        -HmResidualMm $Run.HmResidualMm `
        -Seed 1 `
        -DatasetSeed 1
    Write-Output "[QUEUE] Finished variant: $($Run.Name)"
}

Write-Output "[QUEUE_DONE] Main14 DeepPA follow-up queue completed"
