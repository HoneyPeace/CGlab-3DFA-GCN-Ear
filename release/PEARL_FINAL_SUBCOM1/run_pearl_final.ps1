param(
    [Parameter(Mandatory = $true)]
    [string]$Stage1CheckpointPath,
    [string]$OutputRoot = ".\results\PEARL_FINAL",
    [string]$DataRoot = "..\data",
    [string]$PythonExe = "python",
    [switch]$EvalBest
)

$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Stage1CheckpointPath = (Resolve-Path $Stage1CheckpointPath).Path
if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot = Join-Path $RepoDir $OutputRoot
}
if (-not [System.IO.Path]::IsPathRooted($DataRoot)) {
    $DataRoot = Join-Path $RepoDir $DataRoot
}
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$DataRoot = [System.IO.Path]::GetFullPath($DataRoot)
$ExpName = "PEARL_FINAL_13948"
$UserTag = "pearl_final_seed6"

$commonArgs = @(
    "--exp_name", $ExpName,
    "--output_root", $OutputRoot,
    "--data_root", $DataRoot,
    "--train_dataset_name", "train",
    "--val_dataset_name", "valiation",
    "--test_dataset_name", "test",
    "--val_partition", "val",
    "--Eval_DataType", "test",
    "--seed", "6",
    "--dataset_seed", "1",
    "--epochs", "500",
    "--train_len", "200",
    "--batch_size", "4",
    "--test_batch_size", "1",
    "--num_points", "8192",
    "--sample_way", "FPS",
    "--geom_batch_size", "16",
    "--sigma", "2.5",
    "--k", "30",
    "--plane_knn", "5",
    "--curv_knn", "30",
    "--regression_point_num", "10",
    "--in_channels", "7",
    "--stage1_paconv_source", "original_github",
    "--stage1_in_channels", "3",
    "--stage1_use_cuda_extension", "True",
    "--stage1_checkpoint_path", $Stage1CheckpointPath,
    "--paconv_feature_mode", "center_geometry",
    "--deeppa_feature_mode", "surface_pair_no_delta",
    "--paconv_heatmap_activation_mode", "raw",
    "--eval_heatmap_coord_method", "mds",
    "--heatmap_loss_mode", "adaptive_wing",
    "--calc_scores", "softmax",
    "--ablation_only", "frozen_aux_drop",
    "--latent_fusion_mode", "residual",
    "--fusion_residual_base", "prior",
    "--aux_drop_epochs", "30",
    "--use_stagewise_aux_hm", "True",
    "--stage_downsample_method", "fps",
    "--stage_grid_search_iters", "8",
    "--decoder_fusion", "prog_half_final320",
    "--train_coord_readout", "heatmap_attn_residual",
    "--hm_attn_residual_max_mm", "0.0",
    "--heatmap_activation_mode", "sigmoid",
    "--coord_loss_mode", "focal_l1",
    "--coord_term_weight", "1.0",
    "--struct_loss_mode", "coord",
    "--struct_term_weight", "1.0",
    "--surface_loss_mode", "topk",
    "--surface_term_weight", "1.0",
    "--curv_alpha", "10.0",
    "--dir_beta", "1.0",
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
    "--need_resample", "False",
    "--model", "frozen_aux_drop",
    "--user_tag", $UserTag,
    "--latent_injection_type", "raw"
)

Push-Location $RepoDir
try {
    if ($EvalBest) {
        & $PythonExe "eval.py" @commonArgs `
            "--run_id" "1" `
            "--model_epoch" "Frozen_Aux_Drop_best.t7" `
            "--eval_result_tag" "best" `
            "--heatmap_save_every_n" "0" `
            "--save_per_landmark_heatmap_png" "False"
    } else {
        & $PythonExe "train.py" @commonArgs "--model_epoch" "Single_PAConv_last.t7"
    }
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
