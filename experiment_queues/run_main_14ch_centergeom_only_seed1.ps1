$ErrorActionPreference = "Stop"

# Main-computer default run:
# - raw NPY input: 7ch = XYZ + principal direction 3ch + curvature 1ch
# - PAConv internal edge feature: center_geometry 14ch
# - DeepPA internal local feature: center_geometry 14ch
& "$PSScriptRoot\run_paconv_deeppa_featuremode_seed1.ps1" `
    -PaconvFeatureMode center_geometry `
    -DeepPAFeatureMode center_geometry `
    -RunSuffix base_auxdrop_plateau_fps_hmr2_seed1 `
    -Stage1UserTag main14_center_stage1_seed1 `
    -AblationOnly frozen_aux_drop `
    -StageDownsampleMethod fps `
    -LossSchedule train_hm_plateau `
    -HmResidualMm 2.0 `
    -Seed 1 `
    -DatasetSeed 1
