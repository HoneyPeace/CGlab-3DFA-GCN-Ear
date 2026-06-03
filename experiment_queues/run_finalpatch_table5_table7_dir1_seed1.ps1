$ErrorActionPreference = "Stop"

$RepoDir = "C:\Users\CGlab\Desktop\Earlandmark\Ear_3DFA-GCN"
$WorkspaceRoot = Split-Path -Parent $RepoDir
$PaconvResultRoot = Join-Path $WorkspaceRoot "results\MainPAConv_DeepPAReady"
$PaconvExpName = "S2G_MainPAConv_OrigXYZ3_Lat1344_MDS_S1_FinalPatch"
$PaconvTag = "mainpaconv_orig_xyz3_lat1344_mds_s1_finalpatch"

$ResultRoot = Join-Path $WorkspaceRoot "results\OPD18_FinalPatch_T57D1"
$ExpName = "S2G_T57D1_FinalPatch_S1"
$DebugDir = Join-Path $RepoDir "debug_outputs"
$StatusPath = Join-Path $DebugDir "finalpatch_table5_table7_dir1_seed1.status.tsv"
$PidPath = Join-Path $DebugDir "finalpatch_table5_table7_dir1_seed1.pid"
$WaitPidPath = Join-Path $DebugDir "finalpatch_dp18_dir1_seed1.pid"
$CondaBat = "C:\Users\CGlab\anaconda3\condabin\conda.bat"
$VsDevCmd = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat"
$CondaEnv = "EarLandMarking_CGLAB"
$TorchExtDir = Join-Path $DebugDir "torch_ext_finalpatch_table5_table7_dir1_sm86"
$Checkpoint = ""

New-Item -ItemType Directory -Force -Path $DebugDir, $ResultRoot | Out-Null
$PID | Set-Content -LiteralPath $PidPath -Encoding ASCII
"time`tphase`ttable`trow`ttag`tcurv_alpha`tdir_beta`tstatus`tnote" | Set-Content -LiteralPath $StatusPath -Encoding UTF8

function Add-Status {
    param(
        [string]$Phase,
        [string]$Table,
        [string]$Row,
        [string]$Tag,
        [string]$Status,
        [string]$Note
    )
    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $StatusPath -Encoding UTF8 -Value "$now`t$Phase`t$Table`t$Row`t$Tag`t10.0`t1.0`t$Status`t$Note"
}

function Quote-CmdArg {
    param([string]$Value)
    return '"' + $Value.Replace('"', '\"') + '"'
}

function Invoke-CmdLogged {
    param([string]$Name, [string[]]$CmdArgs)
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $outLog = Join-Path $DebugDir "run_${Name}_${stamp}.out.log"
    $errLog = Join-Path $DebugDir "run_${Name}_${stamp}.err.log"
    $summaryLog = Join-Path $DebugDir "run_${Name}_${stamp}.summary.log"
    $argText = ($CmdArgs | ForEach-Object { Quote-CmdArg $_ }) -join " "
    $cmdLine = "call `"$CondaBat`" activate $CondaEnv && call `"$VsDevCmd`" -arch=amd64 -host_arch=amd64 && cd /d `"$RepoDir`" && set `"PYTHONIOENCODING=utf-8`" && set `"PYTHONUTF8=1`" && set `"PYTHONHASHSEED=1`" && set `"TORCH_CUDA_ARCH_LIST=8.6`" && set `"TORCH_EXTENSIONS_DIR=$TorchExtDir`" && python $argText"

    "[Working Directory]`n$RepoDir`n`n[Command]`n$cmdLine`n`n[StdOut]`n$outLog`n`n[StdErr]`n$errLog" |
        Set-Content -LiteralPath $summaryLog -Encoding UTF8

    $proc = Start-Process -FilePath cmd.exe `
        -ArgumentList @("/d", "/c", $cmdLine) `
        -WorkingDirectory $RepoDir `
        -RedirectStandardOutput $outLog `
        -RedirectStandardError $errLog `
        -WindowStyle Hidden `
        -PassThru `
        -Wait

    if ($proc.ExitCode -ne 0) {
        throw "Run failed: $Name. Inspect $outLog and $errLog"
    }
}

function Wait-For-PidFile {
    param([string]$Path)
    Add-Status "wait" "-" "-" "-" "START" "Waiting for baseline DeepPA pid file: $Path"
    while (-not (Test-Path -LiteralPath $Path)) {
        Start-Sleep -Seconds 30
    }

    $pidText = (Get-Content -LiteralPath $Path -Raw).Trim()
    if ($pidText -eq "") {
        throw "Empty wait pid file: $Path"
    }

    $waitPid = [int]$pidText
    Add-Status "wait" "-" "-" "-" "RUNNING" "Waiting for baseline DeepPA PID $waitPid"
    while (Get-Process -Id $waitPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 60
    }
    Add-Status "wait" "-" "-" "-" "COMPLETE" "Baseline DeepPA PID $waitPid exited"
}

function Find-PaconvCheckpoint {
    $expDir = Join-Path $PaconvResultRoot $PaconvExpName
    if (-not (Test-Path -LiteralPath $expDir)) {
        throw "PAConv exp dir not found: $expDir"
    }
    $runDir = Get-ChildItem -LiteralPath $expDir -Directory -Filter "*${PaconvTag}*" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $runDir) {
        throw "PAConv run dir not found under: $expDir"
    }
    $checkpointPath = Join-Path $runDir.FullName "models\Original_PAConv_last.t7"
    if (-not (Test-Path -LiteralPath $checkpointPath)) {
        throw "PAConv checkpoint not found: $checkpointPath"
    }
    return $checkpointPath
}

function Get-EvalUserTagFromRunDir {
    param([string]$RunDirName)
    $prefix = "FPS8192_sigma2.5_batch4_train200_"
    $evalUserTag = $RunDirName
    if ($evalUserTag.StartsWith($prefix)) {
        $evalUserTag = $evalUserTag.Substring($prefix.Length)
    }
    return ($evalUserTag -replace "_\d+$", "")
}

function Find-RunDir {
    param([string]$Tag)
    $expDir = Join-Path $ResultRoot $ExpName
    if (-not (Test-Path -LiteralPath $expDir)) { return $null }
    return Get-ChildItem -LiteralPath $expDir -Directory -Filter "*${Tag}*" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

function Build-Args {
    param(
        [string]$Entry,
        [object]$Variant,
        [string]$UserTag,
        [string]$ModelEpoch = "",
        [string]$RunId = "",
        [string]$EvalResultTag = ""
    )

    $args = @(
        $Entry,
        "--exp_name", $ExpName,
        "--user_tag", $UserTag,
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
        "--stage1_paconv_source", "original_github",
        "--stage1_in_channels", "3",
        "--stage1_use_cuda_extension", "True",
        "--stage1_checkpoint_path", $Checkpoint,
        "--paconv_feature_mode", "center_geometry",
        "--deeppa_feature_mode", "surface_pair_no_delta",
        "--paconv_heatmap_activation_mode", "raw",
        "--eval_heatmap_coord_method", "mds",
        "--heatmap_loss_mode", "adaptive_wing",
        "--calc_scores", "softmax",
        "--ablation_only", $Variant.AblationOnly,
        "--latent_injection_type", "raw",
        "--latent_fusion_mode", $Variant.LatentFusion,
        "--fusion_residual_base", "prior",
        "--aux_drop_epochs", "30",
        "--use_stagewise_aux_hm", "True",
        "--stage_downsample_method", "fps",
        "--stage_grid_search_iters", "8",
        "--decoder_fusion", $Variant.DecoderFusion,
        "--train_coord_readout", $Variant.CoordReadout,
        "--hm_attn_residual_max_mm", "0.0",
        "--heatmap_activation_mode", "sigmoid",
        "--coord_loss_mode", "focal_l1",
        "--struct_loss_mode", "coord",
        "--surface_loss_mode", "topk",
        "--curv_alpha", "10.0",
        "--dir_beta", "1.0",
        "--loss_schedule", $Variant.LossSchedule,
        "--fixed_main_weight", "1.0",
        "--fixed_geom_weight", "1.0",
        "--model", $Variant.Model
    )

    if ($Variant.LossSchedule -eq "train_hm_plateau") {
        $args += @(
            "--fixed_heatmap_epochs", "60",
            "--plateau_start_epoch", "30",
            "--plateau_window", "20",
            "--plateau_patience", "10",
            "--plateau_threshold", "0.01",
            "--plateau_transition_epochs", "1",
            "--plateau_transition_mode", "linear"
        )
    }
    if ($ModelEpoch -ne "") { $args += @("--model_epoch", $ModelEpoch) }
    if ($RunId -ne "") { $args += @("--run_id", $RunId) }
    if ($EvalResultTag -ne "") { $args += @("--eval_result_tag", $EvalResultTag) }
    return $args
}

$variants = @(
    [pscustomobject]@{
        Table = "Table5"; Row = "Downscaling Add / Upscaling Fusion"; Tag = "t5_add_fus_d1_s1";
        AblationOnly = "frozen_aux_drop"; Model = "frozen_aux_drop"; LastCkpt = "Frozen_Aux_Drop_last.t7"; BestCkpt = "Frozen_Aux_Drop_best.t7";
        LatentFusion = "add"; DecoderFusion = "prog_half_final320"; CoordReadout = "heatmap_attn_residual"; LossSchedule = "train_hm_plateau"
    },
    [pscustomobject]@{
        Table = "Table5"; Row = "Downscaling Concat / Upscaling Fusion"; Tag = "t5_cat_fus_d1_s1";
        AblationOnly = "frozen_aux_drop"; Model = "frozen_aux_drop"; LastCkpt = "Frozen_Aux_Drop_last.t7"; BestCkpt = "Frozen_Aux_Drop_best.t7";
        LatentFusion = "concat"; DecoderFusion = "prog_half_final320"; CoordReadout = "heatmap_attn_residual"; LossSchedule = "train_hm_plateau"
    },
    [pscustomobject]@{
        Table = "Table5"; Row = "Downscaling Residual / Upscaling Add"; Tag = "t5_res_add_d1_s1";
        AblationOnly = "frozen_aux_drop"; Model = "frozen_aux_drop"; LastCkpt = "Frozen_Aux_Drop_last.t7"; BestCkpt = "Frozen_Aux_Drop_best.t7";
        LatentFusion = "residual"; DecoderFusion = "add"; CoordReadout = "heatmap_attn_residual"; LossSchedule = "train_hm_plateau"
    },
    [pscustomobject]@{
        Table = "Table5"; Row = "Downscaling Residual / Upscaling Concat"; Tag = "t5_res_cat_d1_s1";
        AblationOnly = "frozen_aux_drop"; Model = "frozen_aux_drop"; LastCkpt = "Frozen_Aux_Drop_last.t7"; BestCkpt = "Frozen_Aux_Drop_best.t7";
        LatentFusion = "residual"; DecoderFusion = "raw_concat"; CoordReadout = "heatmap_attn_residual"; LossSchedule = "train_hm_plateau"
    },
    [pscustomobject]@{
        Table = "Table7"; Row = "Plateau X / Aux X / HMR X"; Tag = "t7_x_x_x_d1_s1";
        AblationOnly = "frozen_no_aux"; Model = "frozen_no_aux"; LastCkpt = "Frozen_No_Aux_last.t7"; BestCkpt = "Frozen_No_Aux_best.t7";
        LatentFusion = "residual"; DecoderFusion = "prog_half_final320"; CoordReadout = "topk"; LossSchedule = "val_adaptive"
    },
    [pscustomobject]@{
        Table = "Table7"; Row = "Plateau O / Aux X / HMR X"; Tag = "t7_o_x_x_d1_s1";
        AblationOnly = "frozen_no_aux"; Model = "frozen_no_aux"; LastCkpt = "Frozen_No_Aux_last.t7"; BestCkpt = "Frozen_No_Aux_best.t7";
        LatentFusion = "residual"; DecoderFusion = "prog_half_final320"; CoordReadout = "topk"; LossSchedule = "train_hm_plateau"
    },
    [pscustomobject]@{
        Table = "Table7"; Row = "Plateau O / Aux O / HMR X"; Tag = "t7_o_o_x_d1_s1";
        AblationOnly = "frozen_aux_drop"; Model = "frozen_aux_drop"; LastCkpt = "Frozen_Aux_Drop_last.t7"; BestCkpt = "Frozen_Aux_Drop_best.t7";
        LatentFusion = "residual"; DecoderFusion = "prog_half_final320"; CoordReadout = "topk"; LossSchedule = "train_hm_plateau"
    },
    [pscustomobject]@{
        Table = "Table7"; Row = "Plateau O / Aux X / HMR O"; Tag = "t7_o_x_o_d1_s1";
        AblationOnly = "frozen_no_aux"; Model = "frozen_no_aux"; LastCkpt = "Frozen_No_Aux_last.t7"; BestCkpt = "Frozen_No_Aux_best.t7";
        LatentFusion = "residual"; DecoderFusion = "prog_half_final320"; CoordReadout = "heatmap_attn_residual"; LossSchedule = "train_hm_plateau"
    }
)

try {
    Add-Status "queue" "-" "-" "-" "START" "PID=$PID; waits for finalpatch DeepPA then runs Table5/Table7 alpha+beta"
    Wait-For-PidFile -Path $WaitPidPath
    $Checkpoint = Find-PaconvCheckpoint
    Add-Status "checkpoint" "-" "-" "-" "READY" $Checkpoint

    foreach ($variant in $variants) {
        Add-Status "train" $variant.Table $variant.Row $variant.Tag "START" "Stage1=$Checkpoint"
        $runName = (($variant.Tag.ToUpper() -replace "[^A-Z0-9_]", "_") + "_FINALPATCH")
        Invoke-CmdLogged -Name $runName -CmdArgs (Build-Args -Entry "run_frozen.py" -Variant $variant -UserTag $variant.Tag)

        $runDir = Find-RunDir -Tag $variant.Tag
        if ($null -eq $runDir) {
            throw "Run dir not found for $($variant.Tag)"
        }
        $evalUserTag = Get-EvalUserTagFromRunDir -RunDirName $runDir.Name

        Add-Status "eval" $variant.Table $variant.Row $variant.Tag "START_LAST" $runDir.FullName
        Invoke-CmdLogged -Name ($runName + "_EVAL_LAST") -CmdArgs (Build-Args -Entry "eval.py" -Variant $variant -UserTag $evalUserTag -ModelEpoch $variant.LastCkpt -RunId "1" -EvalResultTag "last")

        if (Test-Path -LiteralPath (Join-Path $runDir.FullName ("models\" + $variant.BestCkpt))) {
            Add-Status "eval" $variant.Table $variant.Row $variant.Tag "START_BEST" $runDir.FullName
            Invoke-CmdLogged -Name ($runName + "_EVAL_BEST") -CmdArgs (Build-Args -Entry "eval.py" -Variant $variant -UserTag $evalUserTag -ModelEpoch $variant.BestCkpt -RunId "1" -EvalResultTag "best")
        } else {
            Add-Status "eval" $variant.Table $variant.Row $variant.Tag "NO_BEST" "$($variant.BestCkpt) not found"
        }

        Add-Status "queue" $variant.Table $variant.Row $variant.Tag "READY_FOR_COMPARE" $runDir.FullName
    }

    Add-Status "queue" "-" "-" "-" "COMPLETE" $ResultRoot
} catch {
    Add-Status "queue" "-" "-" "-" "FAILED" $_.Exception.Message
    throw
}
