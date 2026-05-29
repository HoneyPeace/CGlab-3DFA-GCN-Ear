$ErrorActionPreference = "Stop"

$RepoDir = "C:\Users\CGlab\Desktop\Earlandmark\Ear_3DFA-GCN"
$ResultRoot = "C:\Users\CGlab\Desktop\Earlandmark\results\D_ConvNbrAttrScore14_DP14_BoundarySweep_Main"
$ExpName = "S2G_DConvNbrAttrScore14_DP14_BoundSweep_Seed1"
$Checkpoint = "C:\Users\CGlab\Desktop\Earlandmark\results\MainPAConv_DeepPAReady\S2G_MainPAConv_Raw7_ConvNbrAttrScore14Orig_MDS_B4_Seed1\FPS8192_sigma2.5_batch4_train200_mainpaconv_raw7_convnbrattr_score14orig_b4_mds_seed1_1\models\Single_PAConv_last.t7"
$DebugDir = Join-Path $RepoDir "debug_outputs"
$StatusPath = Join-Path $DebugDir "d_convnbrattr_dp14_boundary_sweep_after_xlna_seed1.status.tsv"
$CurrentQueuePidPath = Join-Path $DebugDir "main_opxyz_xyzlocal_neighbor_attr_boundary_sweep_seed1.pid"
$CondaBat = "C:\Users\CGlab\anaconda3\condabin\conda.bat"
$VsDevCmd = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat"
$CondaEnv = "EarLandMarking_CGLAB"
$TorchExtDir = Join-Path $DebugDir "torch_ext_d_convnbrattr_dp14_bound_sweep_sm86"

New-Item -ItemType Directory -Force -Path $DebugDir | Out-Null
New-Item -ItemType Directory -Force -Path $ResultRoot | Out-Null
"time`tphase`ttag`tsetting`tstatus`tnote" | Set-Content -Path $StatusPath -Encoding UTF8

function Add-Status {
    param(
        [string]$Phase,
        [string]$Tag,
        [string]$Setting,
        [string]$Status,
        [string]$Note
    )
    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $StatusPath -Encoding UTF8 -Value "$now`t$Phase`t$Tag`t$Setting`t$Status`t$Note"
}

function Quote-CmdArg {
    param([string]$Value)
    return '"' + $Value.Replace('"', '\"') + '"'
}

function Invoke-CmdLogged {
    param(
        [string]$Name,
        [string[]]$Args
    )
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $outLog = Join-Path $DebugDir "run_${Name}_${stamp}.out.log"
    $errLog = Join-Path $DebugDir "run_${Name}_${stamp}.err.log"
    $summaryLog = Join-Path $DebugDir "run_${Name}_${stamp}.summary.log"
    $argText = ($Args | ForEach-Object { Quote-CmdArg $_ }) -join " "
    $cmdLine = "call `"$CondaBat`" activate $CondaEnv && call `"$VsDevCmd`" -arch=amd64 -host_arch=amd64 && cd /d `"$RepoDir`" && set `"PYTHONIOENCODING=utf-8`" && set `"PYTHONUTF8=1`" && set `"PYTHONHASHSEED=1`" && set `"TORCH_CUDA_ARCH_LIST=8.6`" && set `"TORCH_EXTENSIONS_DIR=$TorchExtDir`" && python $argText"

    "[Working Directory]`n$RepoDir`n`n[Command]`n$cmdLine`n`n[StdOut]`n$outLog`n`n[StdErr]`n$errLog" |
        Set-Content -Path $summaryLog -Encoding UTF8

    cmd.exe /d /c $cmdLine > $outLog 2> $errLog
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed: $Name. Inspect $outLog and $errLog"
    }
    return @{
        OutLog = $outLog
        ErrLog = $errLog
        SummaryLog = $summaryLog
    }
}

function Build-CommonArgs {
    param(
        [string]$Entry,
        [string]$Tag,
        [string]$Bound,
        [string]$ModelEpoch = "",
        [string]$RunId = ""
    )

    $args = @(
        $Entry,
        "--exp_name", $ExpName,
        "--user_tag", $Tag,
        "--output_root", $ResultRoot,
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
        "--stage1_checkpoint_path", $Checkpoint,
        "--paconv_feature_mode", "center_geometry",
        "--paconv_conv_feature_mode", "neighbor_attr",
        "--paconv_scorenet_feature_mode", "original_center_geometry",
        "--deeppa_feature_mode", "center_geometry",
        "--paconv_heatmap_activation_mode", "raw",
        "--eval_heatmap_coord_method", "mds",
        "--heatmap_loss_mode", "adaptive_wing",
        "--calc_scores", "softmax",
        "--ablation_only", "frozen_aux_drop",
        "--latent_injection_type", "raw",
        "--fusion_residual_base", "prior",
        "--aux_drop_epochs", "30",
        "--use_stagewise_aux_hm", "True",
        "--stage_downsample_method", "fps",
        "--stage_grid_search_iters", "8",
        "--decoder_fusion", "prog_half_final320",
        "--train_coord_readout", "heatmap_attn_residual",
        "--hm_attn_residual_max_mm", $Bound,
        "--heatmap_activation_mode", "sigmoid",
        "--coord_loss_mode", "focal_l1",
        "--struct_loss_mode", "coord",
        "--surface_loss_mode", "topk",
        "--curv_alpha", "10.0",
        "--dir_beta", "0.0",
        "--loss_schedule", "train_hm_plateau",
        "--fixed_heatmap_epochs", "60",
        "--plateau_start_epoch", "30",
        "--plateau_window", "20",
        "--plateau_patience", "10",
        "--plateau_threshold", "0.01",
        "--plateau_transition_epochs", "1",
        "--plateau_transition_mode", "linear",
        "--fixed_main_weight", "1.0",
        "--fixed_geom_weight", "1.0",
        "--model", "frozen_aux_drop"
    )

    if ($ModelEpoch -ne "") {
        $args += @("--model_epoch", $ModelEpoch)
    }
    if ($RunId -ne "") {
        $args += @("--run_id", $RunId)
    }
    return $args
}

function Find-RunDir {
    param([string]$Tag)
    $expDir = Join-Path $ResultRoot $ExpName
    if (-not (Test-Path $expDir)) {
        return $null
    }
    return Get-ChildItem -Path $expDir -Directory -Filter "*${Tag}*" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

function Has-ResultArtifacts {
    param([System.IO.DirectoryInfo]$RunDir)
    if ($null -eq $RunDir) {
        return $false
    }
    $txt = Get-ChildItem -Path $RunDir.FullName -Recurse -Filter "*Results*.txt" -ErrorAction SilentlyContinue | Select-Object -First 1
    $xlsx = Get-ChildItem -Path $RunDir.FullName -Recurse -Filter "*Results*.xlsx" -ErrorAction SilentlyContinue | Select-Object -First 1
    return ($null -ne $txt -and $null -ne $xlsx)
}

if (-not (Test-Path $Checkpoint)) {
    throw "Stage1 checkpoint not found: $Checkpoint"
}

if (Test-Path $CurrentQueuePidPath) {
    $currentPidText = (Get-Content -Path $CurrentQueuePidPath -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($currentPidText -match "^\d+$") {
        $currentPid = [int]$currentPidText
        $currentProc = Get-Process -Id $currentPid -ErrorAction SilentlyContinue
        if ($null -ne $currentProc) {
            Add-Status "wait" "main_opxyz_xyzlocal_neighbor_attr_boundary_sweep_seed1" "existing_queue" "WAIT" "Waiting for PID $currentPid to exit before using GPU"
            Wait-Process -Id $currentPid
            Add-Status "wait" "main_opxyz_xyzlocal_neighbor_attr_boundary_sweep_seed1" "existing_queue" "DONE" "Previous queue exited"
        }
    }
}

$bounds = @(
    @{ Tag = "d_convnbrattr_dp14_hmr0p5_seed1"; Name = "HMR0P5"; Bound = "0.5" },
    @{ Tag = "d_convnbrattr_dp14_hmr1p0_seed1"; Name = "HMR1P0"; Bound = "1.0" },
    @{ Tag = "d_convnbrattr_dp14_hmr1p5_seed1"; Name = "HMR1P5"; Bound = "1.5" },
    @{ Tag = "d_convnbrattr_dp14_hmr2p0_seed1"; Name = "HMR2P0"; Bound = "2.0" }
)

foreach ($item in $bounds) {
    $tag = $item.Tag
    $bound = $item.Bound
    $name = "D_CONVNBRATTR_DP14_" + $item.Name

    Add-Status "train" $tag $bound "START" $name
    Invoke-CmdLogged -Name $name -Args (Build-CommonArgs -Entry "run_frozen.py" -Tag $tag -Bound $bound) | Out-Null

    $runDir = Find-RunDir -Tag $tag
    Add-Status "eval" $tag $bound "START" $(if ($null -ne $runDir) { $runDir.FullName } else { "run dir pending" })
    Invoke-CmdLogged -Name ($name + "_eval") -Args (Build-CommonArgs -Entry "eval.py" -Tag $tag -Bound $bound -ModelEpoch "Frozen_Aux_Drop_last.t7" -RunId "1") | Out-Null

    $runDir = Find-RunDir -Tag $tag
    if (-not (Has-ResultArtifacts -RunDir $runDir)) {
        throw "Missing Results txt/xlsx for $tag"
    }
    Add-Status "queue" $tag $bound "READY_FOR_NOTION" $runDir.FullName
}

Add-Status "queue" "d_convnbrattr_dp14_boundary_sweep" "all" "COMPLETE" $ResultRoot
