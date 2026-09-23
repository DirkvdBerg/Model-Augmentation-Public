<#
watchdog.ps1  -  launch a command, sample its resources, kill it before the machine runs dry.

Usage (from the repo root, see telica-real/README.md):
  powershell -NoProfile -ExecutionPolicy Bypass -File telica-real/tools/watchdog.ps1 `
      -Run "conda run --no-capture-output -n GraduationProject python -u telica-real/x.py" `
      -Out telica-real/outputs/<run>

Every IntervalS seconds it appends one row to <Out>/resources.csv:
  process-tree working set, tree CPU (% of all logical cores), system available RAM, C: free.
Thresholds (TR-002):
  WATCHDOG ALERT  available RAM < RamAlertGB (4.0)  or  C: free < DiskAlertGB (3.0)
  WATCHDOG KILL   available RAM < RamKillGB  (2.5)  or  C: free < DiskKillGB  (2.0)
  -> taskkill /T /F on the whole tree.
The child inherits stdout/stderr, so its output streams live into the caller's log.
Ends with one line starting WATCHDOG SUMMARY (also written to <Out>/watchdog_summary.txt).
#>
param(
    [Parameter(Mandatory = $true)][string]$Run,
    [Parameter(Mandatory = $true)][string]$Out,
    [double]$RamKillGB = 2.5,
    [double]$DiskKillGB = 2.0,
    [double]$RamAlertGB = 4.0,
    [double]$DiskAlertGB = 3.0,
    [int]$IntervalS = 5,
    [string]$Drive = 'C',
    [double]$MinKillS = 0    # test hook (G0): ignore kill conditions before this many seconds
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$csv = Join-Path $Out 'resources.csv'
'time_iso,elapsed_s,tree_n,tree_ws_mb,tree_cpu_pct,avail_ram_gb,disk_free_gb,flag' |
    Out-File -FilePath $csv -Encoding ascii

# Child environment (TR-002): live output, bounded thread pools.
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'

function Get-TreeProcs([int]$rootId) {
    $all = @(Get-CimInstance Win32_Process -Property ProcessId, ParentProcessId, WorkingSetSize, KernelModeTime, UserModeTime)
    $ids = New-Object 'System.Collections.Generic.List[int]'
    $ids.Add($rootId)
    $i = 0
    while ($i -lt $ids.Count) {
        $parent = $ids[$i]
        foreach ($q in $all) {
            $qid = [int]$q.ProcessId
            if ([int]$q.ParentProcessId -eq $parent -and $qid -ne $parent -and -not $ids.Contains($qid)) {
                $ids.Add($qid)
            }
        }
        $i++
    }
    return @($all | Where-Object { $ids.Contains([int]$_.ProcessId) })
}

function Get-AvailRamGB {
    $os = Get-CimInstance Win32_OperatingSystem -Property FreePhysicalMemory
    return [double]$os.FreePhysicalMemory / 1MB    # KB -> GB
}

function Get-DiskFreeGB {
    return [double](Get-PSDrive $Drive).Free / 1GB
}

$nLogical = [Environment]::ProcessorCount
$start = Get-Date
Write-Output ("[wd] launch  out=" + $Out)
Write-Output ("[wd] command " + $Run)
$p = Start-Process -FilePath 'cmd.exe' -ArgumentList ('/s /c "' + $Run + '"') -NoNewWindow -PassThru
$null = $p.Handle    # cache the handle so ExitCode is readable after exit
$rootId = $p.Id

$peakWs = 0.0; $minRam = 1e9; $minDisk = 1e9
$prevCpu = $null; $prevT = $null
$killed = $false; $killReason = ''
$alertRam = $false; $alertDisk = $false
$nSamples = 0

while ($true) {
    $now = Get-Date
    $el = ($now - $start).TotalSeconds
    $procs = @()
    try { $procs = Get-TreeProcs $rootId } catch { $procs = @() }
    $ws = 0.0; $cpuTicks = 0.0
    foreach ($q in $procs) {
        $ws += [double]$q.WorkingSetSize
        $cpuTicks += [double]$q.KernelModeTime + [double]$q.UserModeTime
    }
    $wsMb = $ws / 1MB
    $cpuPct = 0.0
    if ($null -ne $prevCpu -and $el -gt $prevT) {
        # 100 ns ticks -> s, divided by wall time and core count
        $cpuPct = [math]::Max(0.0, ($cpuTicks - $prevCpu) / 1e7 / ($el - $prevT) / $nLogical * 100.0)
    }
    $prevCpu = $cpuTicks; $prevT = $el
    $ram = Get-AvailRamGB
    $disk = Get-DiskFreeGB
    if ($wsMb -gt $peakWs) { $peakWs = $wsMb }
    if ($ram -lt $minRam) { $minRam = $ram }
    if ($disk -lt $minDisk) { $minDisk = $disk }

    $flag = ''
    if ($ram -lt $RamAlertGB) {
        $flag = 'ram_alert'
        if (-not $alertRam) {
            Write-Output ('WATCHDOG ALERT available RAM {0:N2} GB < {1} GB (tree {2:N0} MB)' -f $ram, $RamAlertGB, $wsMb)
            $alertRam = $true
        }
    } else { $alertRam = $false }
    if ($disk -lt $DiskAlertGB) {
        $flag = ($flag + ' disk_alert').Trim()
        if (-not $alertDisk) {
            Write-Output ('WATCHDOG ALERT {0}: free {1:N2} GB < {2} GB' -f $Drive, $disk, $DiskAlertGB)
            $alertDisk = $true
        }
    } else { $alertDisk = $false }

    $kill = $false
    if ($ram -lt $RamKillGB) { $kill = $true; $killReason = ('available RAM {0:N2} GB < {1} GB' -f $ram, $RamKillGB) }
    elseif ($disk -lt $DiskKillGB) { $kill = $true; $killReason = ('{0}: free {1:N2} GB < {2} GB' -f $Drive, $disk, $DiskKillGB) }
    if ($kill -and $el -lt $MinKillS) { $kill = $false; $killReason = '' }
    if ($kill) { $flag = ($flag + ' kill').Trim() }

    ('{0},{1:F1},{2},{3:F1},{4:F1},{5:F3},{6:F3},{7}' -f $now.ToString('s'), $el, $procs.Count, $wsMb, $cpuPct, $ram, $disk, $flag) |
        Out-File -FilePath $csv -Encoding ascii -Append
    $nSamples++

    if ($kill -and -not $p.HasExited) {
        Write-Output ('WATCHDOG KILL {0}; killing tree of PID {1} ({2} processes, {3:N0} MB)' -f $killReason, $rootId, $procs.Count, $wsMb)
        & taskkill.exe /T /F /PID $rootId 2>&1 | Out-Null
        $killed = $true
        Start-Sleep -Seconds 1
        break
    }
    if ($nSamples % 12 -eq 0) {
        Write-Output ('[wd] t={0:N0}s tree={1:N0} MB cpu={2:N0}% availRAM={3:N2} GB {4}:free={5:N2} GB' -f $el, $wsMb, $cpuPct, $ram, $Drive, $disk)
    }
    if ($p.WaitForExit($IntervalS * 1000)) { break }
}

$exitCode = 'killed'
if (-not $killed) {
    $p.WaitForExit()
    $exitCode = $p.ExitCode
}
$dur = ((Get-Date) - $start).TotalSeconds
$summary = ('WATCHDOG SUMMARY exit={0} killed={1} duration_s={2:N1} samples={3} peak_tree_ws_mb={4:N0} min_avail_ram_gb={5:N2} min_{6}_free_gb={7:N2} kill_reason="{8}"' -f $exitCode, $killed, $dur, $nSamples, $peakWs, $minRam, $Drive, $minDisk, $killReason)
Write-Output $summary
$summary | Out-File -FilePath (Join-Path $Out 'watchdog_summary.txt') -Encoding ascii
if ($killed) { exit 137 }
if ($exitCode -is [int]) { exit $exitCode }
exit 0
