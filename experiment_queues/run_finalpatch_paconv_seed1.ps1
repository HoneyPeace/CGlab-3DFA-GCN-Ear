$ErrorActionPreference = "Stop"

$RepoDir = Split-Path -Parent $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $RepoDir
$DebugDir = Join-Path $RepoDir "debug_outputs"
$OutputRoot = Join-Path $WorkspaceRoot "results\MainPAConv_DeepPAReady"
$CondaEnv = "EarLandMarking_CGLAB"
$CondaBat = Join-Path $env:USERPROFILE "anaconda3\condabin\conda.bat"
$VsDevCmd = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat"
$TorchExtDir = Join-Path $DebugDir "torch_ext_finalpatch_paconv_sm86"

if (-not (Test-Path -LiteralPath $CondaBat)) {
    $CondaBat = "C:\Users\CGlab\anaconda3\condabin\conda.bat"
}
if (-not (Test-Path -LiteralPath $CondaBat)) {
    throw "conda.bat not found. Edit `$CondaBat in this script for this machine."
}
if (-not (Test-Path -LiteralPath $VsDevCmd)) {
    throw "VsDevCmd.bat not found. Install VS2019 Build Tools or edit `$VsDevCmd in this script."
}

New-Item -ItemType Directory -Force -Path $DebugDir, $OutputRoot | Out-Null

$RunTag = "mainpaconv_orig_xyz3_lat1344_mds_s1_finalpatch"
$ExpName = "S2G_MainPAConv_OrigXYZ3_Lat1344_MDS_S1_FinalPatch"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$PidPath = Join-Path $DebugDir "finalpatch_paconv_seed1.pid"
$StatusPath = Join-Path $DebugDir "finalpatch_paconv_seed1.status.tsv"
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
    "legacy_3dfa_gcn\original_paconv_current_split.py",
    "--data_root", (Join-Path $WorkspaceRoot "data"),
    "--output_root", $OutputRoot,
    "--exp_name", $ExpName,
    "--user_tag", $RunTag,
    "--train_dataset_name", "train",
    "--val_dataset_name", "valiation",
    "--test_dataset_name", "test",
    "--val_partition", "val",
    "--eval_datatype", "test",
    "--num_points", "8192",
    "--sigma", "2.5",
    "--landmark_num", "36",
    "--epochs", "500",
    "--batch_size", "4",
    "--test_batch_size", "1",
    "--num_workers", "0",
    "--k", "30",
    "--regression_point_num", "10",
    "--seed", "1",
    "--dataset_seed", "1",
    "--val_method", "mds",
    "--use_cuda_extension"
)

$ArgText = Join-CmdArgs $PyArgs
$Cmd = "call `"$CondaBat`" activate $CondaEnv && call `"$VsDevCmd`" -arch=amd64 -host_arch=amd64 && cd /d `"$RepoDir`" && set `"PYTHONIOENCODING=utf-8`" && set `"PYTHONUTF8=1`" && set `"PYTHONHASHSEED=1`" && set `"TORCH_CUDA_ARCH_LIST=8.6`" && set `"TORCH_EXTENSIONS_DIR=$TorchExtDir`" && python $ArgText"

"time`tphase`ttag`tstatus`tnote" | Out-File -LiteralPath $StatusPath -Encoding UTF8
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`tqueue`t$RunTag`tSTART`tOriginal PAConv with patched train landmarks/heatmaps: 004, 069, 100, 115, 119" | Out-File -LiteralPath $StatusPath -Append -Encoding UTF8
"[START] $RunTag $(Get-Date -Format s)" | Out-File -LiteralPath $Summary -Encoding UTF8
"[REPO] $RepoDir" | Out-File -LiteralPath $Summary -Append -Encoding UTF8
"[OUTPUT_ROOT] $OutputRoot" | Out-File -LiteralPath $Summary -Append -Encoding UTF8
"[EXP_NAME] $ExpName" | Out-File -LiteralPath $Summary -Append -Encoding UTF8
"[COMMAND] python $ArgText" | Out-File -LiteralPath $Summary -Append -Encoding UTF8

Write-Output "[QUEUE] Starting $RunTag"
Write-Output "[QUEUE] PID path: $PidPath"
Write-Output "[QUEUE] Status  : $StatusPath"
Write-Output "[QUEUE] Summary : $Summary"
Write-Output "[QUEUE] StdOut  : $StdOut"
Write-Output "[QUEUE] StdErr  : $StdErr"

$Proc = Start-Process -FilePath cmd.exe `
    -ArgumentList @("/d", "/c", $Cmd) `
    -WorkingDirectory $RepoDir `
    -RedirectStandardOutput $StdOut `
    -RedirectStandardError $StdErr `
    -WindowStyle Hidden `
    -PassThru

$Proc.Id | Out-File -LiteralPath $PidPath -Encoding ASCII
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`ttrain_eval`t$RunTag`tRUNNING`tPID=$($Proc.Id); stdout=$StdOut" | Out-File -LiteralPath $StatusPath -Append -Encoding UTF8
Write-Output "[QUEUE] Started PID=$($Proc.Id)"
