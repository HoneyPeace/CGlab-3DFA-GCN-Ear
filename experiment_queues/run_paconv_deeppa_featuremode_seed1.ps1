param(
    [ValidateSet("", "center_geometry", "full_extension")]
    [string]$FeatureMode = "",
    [ValidateSet("", "center_geometry", "full_extension")]
    [string]$PaconvFeatureMode = "",
    [ValidateSet("", "center_geometry", "full_extension")]
    [string]$DeepPAFeatureMode = ""
)

$ErrorActionPreference = "Stop"

$RepoDir = Split-Path -Parent $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $RepoDir
$DebugDir = Join-Path $RepoDir "debug_outputs"
$OutputRootBase = Join-Path $WorkspaceRoot "results\MainPAConv_DeepPAReady"
$CondaEnv = "EarLandMarking_CGLAB"
$CondaBat = Join-Path $env:USERPROFILE "anaconda3\condabin\conda.bat"
$VsDevCmd = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat"

if (-not (Test-Path -LiteralPath $CondaBat)) {
    $CondaBat = "C:\Users\CGlab\anaconda3\condabin\conda.bat"
}
if (-not (Test-Path -LiteralPath $CondaBat)) {
    throw "conda.bat not found. Edit `$CondaBat in this script for this machine."
}
if (-not (Test-Path -LiteralPath $VsDevCmd)) {
    throw "VsDevCmd.bat not found. Install VS2019 Build Tools or edit `$VsDevCmd in this script."
}

if (-not $PaconvFeatureMode) {
    $PaconvFeatureMode = if ($FeatureMode) { $FeatureMode } else { "center_geometry" }
}
if (-not $DeepPAFeatureMode) {
    $DeepPAFeatureMode = "center_geometry"
}

$PaShort = if ($PaconvFeatureMode -eq "full_extension") { "pafull" } else { "pacenter" }
$DpShort = if ($DeepPAFeatureMode -eq "full_extension") { "dpfull" } else { "dpcenter" }
$RunTag = $PaShort + "_" + $DpShort + "_hmr2_seed1"
$ExpName = "S2G_Frozen_HMR_" + $PaShort + "_" + $DpShort + "_Seed1"
$OutputRoot = Join-Path $OutputRootBase ($PaShort + "_" + $DpShort)

New-Item -ItemType Directory -Force -Path $DebugDir, $OutputRoot | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Summary = Join-Path $DebugDir ("run_" + $RunTag + "_" + $Stamp + ".summary.log")
$StdOut = Join-Path $DebugDir ("run_" + $RunTag + "_" + $Stamp + ".out.log")
$StdErr = Join-Path $DebugDir ("run_" + $RunTag + "_" + $Stamp + ".err.log")

function Join-CmdArgs {
    param([string[]]$Items)
    $quoted = @()
    foreach ($item in $Items) {
        if ($item -match '[\s&()"]') {
            $quoted += '"' + ($item -replace '"', '\"') + '"'
        } else {
            $quoted += $item
        }
    }
    return ($quoted -join " ")
}

$PyArgs = @(
    "run_frozen.py",
    "--exp_name", $ExpName,
    "--user_tag", $RunTag,
    "--output_root", $OutputRoot,
    "--train_dataset_name", "train",
    "--val_dataset_name", "valiation",
    "--test_dataset_name", "test",
    "--val_partition", "val",
    "--Eval_DataType", "test",
    "--need_resample", "False",
    "--seed", "1",
    "--dataset_seed", "1",
    "--epochs", "500",
    "--train_len", "200",
    "--batch_size", "4",
    "--test_batch_size", "1",
    "--num_points", "8192",
    "--geom_batch_size", "16",
    "--sigma", "2.5",
    "--k", "30",
    "--regression_point_num", "10",
    "--in_channels", "7",
    "--paconv_feature_mode", $PaconvFeatureMode,
    "--deeppa_feature_mode", $DeepPAFeatureMode,
    "--paconv_heatmap_activation_mode", "raw",
    "--eval_heatmap_coord_method", "mds",
    "--heatmap_loss_mode", "adaptive_wing",
    "--calc_scores", "softmax",
    "--ablation_only", "frozen_aux_drop",
    "--latent_injection_type", "raw",
    "--fusion_residual_base", "prior",
    "--aux_drop_epochs", "30",
    "--use_stagewise_aux_hm", "True",
    "--decoder_fusion", "prog_half_final320",
    "--train_coord_readout", "heatmap_attn_residual",
    "--hm_attn_residual_max_mm", "2.0",
    "--heatmap_activation_mode", "sigmoid",
    "--coord_loss_mode", "focal_l1",
    "--struct_loss_mode", "coord",
    "--surface_loss_mode", "topk",
    "--loss_schedule", "train_hm_plateau",
    "--plateau_start_epoch", "30",
    "--plateau_window", "20",
    "--plateau_patience", "10",
    "--plateau_threshold", "0.01",
    "--plateau_transition_epochs", "1",
    "--plateau_transition_mode", "linear",
    "--fixed_main_weight", "1.0",
    "--fixed_geom_weight", "1.0"
)

$ArgText = Join-CmdArgs $PyArgs
$Cmd = "call `"$CondaBat`" activate $CondaEnv && call `"$VsDevCmd`" -arch=amd64 -host_arch=amd64 && cd /d `"$RepoDir`" && set PYTHONIOENCODING=utf-8 && set PYTHONUTF8=1 && set PYTHONHASHSEED=1 && python $ArgText"

"[START] $RunTag $(Get-Date -Format s)" | Out-File -LiteralPath $Summary -Encoding UTF8
"[REPO] $RepoDir" | Out-File -LiteralPath $Summary -Append -Encoding UTF8
"[OUTPUT_ROOT] $OutputRoot" | Out-File -LiteralPath $Summary -Append -Encoding UTF8
"[PACONV_FEATURE_MODE] $PaconvFeatureMode" | Out-File -LiteralPath $Summary -Append -Encoding UTF8
"[DEEPPA_FEATURE_MODE] $DeepPAFeatureMode" | Out-File -LiteralPath $Summary -Append -Encoding UTF8
"[COMMAND] python $ArgText" | Out-File -LiteralPath $Summary -Append -Encoding UTF8

Write-Output "[QUEUE] Starting $RunTag"
Write-Output "[QUEUE] Summary: $Summary"
Write-Output "[QUEUE] StdOut : $StdOut"
Write-Output "[QUEUE] StdErr : $StdErr"

$Proc = Start-Process -FilePath cmd.exe `
    -ArgumentList @("/d", "/c", $Cmd) `
    -WorkingDirectory $RepoDir `
    -RedirectStandardOutput $StdOut `
    -RedirectStandardError $StdErr `
    -WindowStyle Hidden `
    -PassThru `
    -Wait

if ($Proc.ExitCode -ne 0) {
    "[FAIL] $RunTag exit=$($Proc.ExitCode) $(Get-Date -Format s)" | Out-File -LiteralPath $Summary -Append -Encoding UTF8
    throw "Run failed: $RunTag. Inspect $StdErr"
}

"[DONE] $RunTag $(Get-Date -Format s)" | Out-File -LiteralPath $Summary -Append -Encoding UTF8
Write-Output "[QUEUE] Completed $RunTag"
