$ErrorActionPreference = "Stop"

# Main-computer default run:
# - raw NPY input: 7ch = XYZ + principal direction 3ch + curvature 1ch
# - PAConv internal edge feature: center_geometry 14ch
# - DeepPA internal local feature: center_geometry 14ch
& "$PSScriptRoot\run_paconv_deeppa_featuremode_seed1.ps1" -PaconvFeatureMode center_geometry -DeepPAFeatureMode center_geometry
