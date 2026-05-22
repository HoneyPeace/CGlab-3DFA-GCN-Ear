$ErrorActionPreference = "Stop"

# Main-computer default run:
# - raw NPY input: 7ch = XYZ + principal direction 3ch + curvature 1ch
# - PAConv internal edge feature: center_geometry 14ch
# - DeepPA internal local feature: center_geometry 14ch
# - Stage1 is loaded from the PAConv-only command's latest Single_PAConv_last.t7
$RepoDir = Split-Path -Parent $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $RepoDir
$OutputRootBase = Join-Path $WorkspaceRoot "results\MainPAConv_DeepPAReady"
$PaconvExpDir = Join-Path $OutputRootBase "S2G_MainPAConv_Raw7_CenterGeom_MDS_Seed1"
$PaconvRunTag = "mainpaconv_raw7_centergeom_mds_seed1"

if (-not (Test-Path -LiteralPath $PaconvExpDir)) {
    throw "PAConv result folder not found. Run .\experiment_queues\run_mainpaconv_raw7_centergeom_seed1.ps1 first. Missing: $PaconvExpDir"
}

$PaconvRunDir = Get-ChildItem -LiteralPath $PaconvExpDir -Directory |
    Where-Object {
        $_.Name -like "*$PaconvRunTag*" -and
        (Test-Path -LiteralPath (Join-Path $_.FullName "models\Single_PAConv_last.t7"))
    } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $PaconvRunDir) {
    throw "Single_PAConv_last.t7 not found for $PaconvRunTag. Run the PAConv-only command/eval first."
}

$Stage1CheckpointPath = Join-Path $PaconvRunDir.FullName "models\Single_PAConv_last.t7"

& "$PSScriptRoot\run_paconv_deeppa_featuremode_seed1.ps1" `
    -PaconvFeatureMode center_geometry `
    -DeepPAFeatureMode center_geometry `
    -RunSuffix base_auxdrop_plateau_fps_hmr2_seed1 `
    -Stage1UserTag main14_center_stage1_seed1 `
    -Stage1CheckpointPath $Stage1CheckpointPath `
    -AblationOnly frozen_aux_drop `
    -StageDownsampleMethod fps `
    -LossSchedule train_hm_plateau `
    -HmResidualMm 2.0 `
    -Seed 1 `
    -DatasetSeed 1
