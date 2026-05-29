$ErrorActionPreference = "Stop"

$RepoDir = "C:\Users\CGlab\Desktop\Earlandmark\Ear_3DFA-GCN"
$DebugDir = Join-Path $RepoDir "debug_outputs"
$QueuePidPath = Join-Path $DebugDir "main_opxyz_xyzlocal_neighbor_attr_boundary_sweep_seed1.pid"
$QueueStatusPath = Join-Path $DebugDir "main_opxyz_xyzlocal_neighbor_attr_boundary_sweep_seed1.status.tsv"
$StopStatusPath = Join-Path $DebugDir "stop_xlna_after_bound2p0_seed1.status.tsv"
$Bound2RunDir = "C:\Users\CGlab\Desktop\Earlandmark\results\OPXYZ_XYZLocalNeighborAttr_BoundarySweep_Main\S2G_OPXYZ_XLNA_BoundSweep_Seed1\FPS8192_sigma2.5_batch4_train200_xlna_hmr2p0_seed1_1"

"time`tphase`tstatus`tnote" | Set-Content -Path $StopStatusPath -Encoding UTF8

function Add-Status {
    param([string]$Phase, [string]$Status, [string]$Note)
    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $StopStatusPath -Encoding UTF8 -Value "$now`t$Phase`t$Status`t$Note"
}

function Get-ChildProcessIds {
    param([int]$ParentId)
    $children = Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $ParentId }
    foreach ($child in $children) {
        Get-ChildProcessIds -ParentId ([int]$child.ProcessId)
        [int]$child.ProcessId
    }
}

function Stop-ProcessTree {
    param([int]$RootPid)
    $ids = @(Get-ChildProcessIds -ParentId $RootPid)
    [array]::Reverse($ids)
    foreach ($id in $ids) {
        Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
    }
    Stop-Process -Id $RootPid -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path $QueuePidPath)) {
    Add-Status "init" "NO_PID" "Queue PID file not found"
    exit 0
}

$queuePidText = (Get-Content -Path $QueuePidPath -ErrorAction SilentlyContinue | Select-Object -First 1)
if ($queuePidText -notmatch "^\d+$") {
    Add-Status "init" "BAD_PID" "Queue PID was not numeric: $queuePidText"
    exit 0
}
$queuePid = [int]$queuePidText
Add-Status "watch" "START" "Watching xlna_hmr2p0_seed1 artifacts, queue PID $queuePid"

while ($true) {
    $queueProc = Get-Process -Id $queuePid -ErrorAction SilentlyContinue
    if ($null -eq $queueProc) {
        Add-Status "watch" "QUEUE_EXITED" "Queue exited before stop was needed"
        exit 0
    }

    $statusReady = $false
    if (Test-Path $QueueStatusPath) {
        $statusReady = Select-String -Path $QueueStatusPath -Pattern "xlna_hmr2p0_seed1`t2.0`tREADY_FOR_NOTION" -Quiet
    }

    $txt = $null
    $xlsx = $null
    if (Test-Path $Bound2RunDir) {
        $txt = Get-ChildItem -Path $Bound2RunDir -Recurse -Filter "*Results*.txt" -ErrorAction SilentlyContinue | Select-Object -First 1
        $xlsx = Get-ChildItem -Path $Bound2RunDir -Recurse -Filter "*Results*.xlsx" -ErrorAction SilentlyContinue | Select-Object -First 1
    }

    if ($statusReady -or ($null -ne $txt -and $null -ne $xlsx)) {
        Add-Status "watch" "BOUND2_DONE" "2.0 artifacts are ready; stopping original xlna queue before later bounds"
        Stop-ProcessTree -RootPid $queuePid
        Add-Status "stop" "DONE" "Stopped queue PID $queuePid after 2.0"
        exit 0
    }

    Start-Sleep -Seconds 20
}
