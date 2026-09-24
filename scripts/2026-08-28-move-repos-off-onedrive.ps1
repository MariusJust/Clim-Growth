<#
    Move the two paper repos off OneDrive onto the local disk.

        "...\PhD\Papers\URoot"       ->  C:\dev\URoot
        "...\PhD\Papers\Clim_Growth" ->  C:\dev\Clim_Growth

    Same volume (both on C:), so this is an NTFS directory rename: instant, and
    it needs NO extra free space.

    WHAT BLOCKS THE MOVE (v2 hit both):
      1. Your PowerShell session's current directory being inside a source folder.
         Windows will not move a directory that is any process's CWD. This script
         now cd's to C:\ before touching anything.
      2. OneDrive.exe holding handles on its own sync root. It must be stopped,
         not merely paused.
      3. VS Code, File Explorer windows, Docker, or a running python/latex process
         with a file open in either tree. Close those first.

    HEADS UP: once OneDrive restarts it will see these folders as deleted and
    remove them from the cloud. That is inherent to moving out of the sync root.
    After this, the repos are LOCAL ONLY -- make sure they are pushed to a git
    remote (or still exist on the server) before you rely on it.

    Run from ANYWHERE EXCEPT inside the two source trees, e.g.:
        cd C:\
        powershell -ExecutionPolicy Bypass -File "$env:USERPROFILE\OneDrive - Aarhus universitet\PhD\Papers\Clim_Growth\Paper_1\scripts\2026-08-28-move-repos-off-onedrive.ps1"
#>

$ErrorActionPreference = 'Stop'

$Base = "$env:USERPROFILE\OneDrive - Aarhus universitet\PhD\Papers"
$Jobs = @(
    [pscustomobject]@{ Name='URoot';       Src="$Base\URoot";       Dst='C:\dev\URoot';       PreFiles=0; PreBytes=0 },
    [pscustomobject]@{ Name='Clim_Growth'; Src="$Base\Clim_Growth"; Dst='C:\dev\Clim_Growth'; PreFiles=0; PreBytes=0 }
)

function Get-TreeStats([string]$Path) {
    $f = Get-ChildItem -LiteralPath $Path -Recurse -File -Force -ErrorAction SilentlyContinue
    [pscustomobject]@{
        Files = ($f | Measure-Object).Count
        Bytes = [double](($f | Measure-Object -Property Length -Sum).Sum)
    }
}

# ---- 0. get OUT of the folders we are about to move -----------------------
Write-Host "`n=== 0. Pre-flight ===" -ForegroundColor Cyan
$here = (Get-Location).Path
foreach ($j in $Jobs) {
    if ($here -like "$($j.Src)*") {
        Write-Host "  current directory is inside $($j.Name); moving to C:\" -ForegroundColor Yellow
    }
}
Set-Location 'C:\'
Write-Host "  working directory: $((Get-Location).Path)"

foreach ($j in $Jobs) {
    if (-not (Test-Path -LiteralPath $j.Src)) { throw "Source not found: $($j.Src)" }
    if (Test-Path -LiteralPath $j.Dst)        { throw "Destination exists: $($j.Dst)" }
    if ((Split-Path -Qualifier $j.Src) -ne (Split-Path -Qualifier $j.Dst)) {
        throw "$($j.Name): cross-volume move -- this script assumes same volume."
    }
}
Write-Host "  both moves are same-volume: no extra disk space needed." -ForegroundColor Green

# ---- 1. hydrate -----------------------------------------------------------
Write-Host "`n=== 1. Hydrate (remove cloud placeholders) ===" -ForegroundColor Cyan
foreach ($j in $Jobs) {
    Write-Host "  $($j.Name) ..."
    & attrib.exe +P -U "$($j.Src)\*" /s /d 2>$null | Out-Null
}
Write-Host "  WAIT for the OneDrive tray icon to read 'Up to date' before continuing." -ForegroundColor Yellow
Read-Host  "  Press Enter when OneDrive is idle"

foreach ($j in $Jobs) {
    $s = Get-TreeStats $j.Src
    $j.PreFiles = $s.Files; $j.PreBytes = $s.Bytes
    "  {0,-14} {1,7} files  {2,8:N2} GB" -f $j.Name, $s.Files, ($s.Bytes/1GB)
}

# ---- 2. release locks -----------------------------------------------------
Write-Host "`n=== 2. Release locks ===" -ForegroundColor Cyan
Write-Host "  Close VS Code, any File Explorer window showing these folders, and stop"
Write-Host "  anything running out of them (python, latexmk, Docker builds)."
$od = Get-Process OneDrive -ErrorAction SilentlyContinue
if ($od) {
    Write-Host "  OneDrive.exe is running and holds handles on its sync root." -ForegroundColor Yellow
    if ((Read-Host "  Stop OneDrive now? [y/N]") -match '^[Yy]') {
        $od | Stop-Process -Force
        Start-Sleep -Seconds 3
        Write-Host "  OneDrive stopped." -ForegroundColor Green
    } else {
        Write-Host "  Continuing with OneDrive running -- the move will probably fail." -ForegroundColor Red
    }
} else {
    Write-Host "  OneDrive is not running. Good."
}
Read-Host "  Press Enter to start the move"

# ---- 3. move (with retries) ----------------------------------------------
Write-Host "`n=== 3. Move ===" -ForegroundColor Cyan
foreach ($j in $Jobs) {
    Write-Host "  $($j.Src)"
    Write-Host "    -> $($j.Dst)"
    $done = $false
    for ($try = 1; $try -le 3 -and -not $done; $try++) {
        try {
            New-Item -ItemType Directory -Force -Path 'C:\dev' | Out-Null
            Move-Item -LiteralPath $j.Src -Destination $j.Dst
            $done = $true
            Write-Host "    moved" -ForegroundColor Green
        } catch {
            Write-Host "    attempt $try failed: $($_.Exception.Message)" -ForegroundColor Yellow
            if ($try -lt 3) { Start-Sleep -Seconds 5 }
        }
    }
    if (-not $done) {
        Write-Host "    STILL LOCKED. Find the holder with Resource Monitor:" -ForegroundColor Red
        Write-Host "      resmon.exe  ->  CPU tab  ->  Associated Handles  ->  search '$($j.Name)'" -ForegroundColor DarkGray
        Write-Host "    Then re-run this script." -ForegroundColor DarkGray
    }
}

# ---- 4. verify ------------------------------------------------------------
Write-Host "`n=== 4. Verify ===" -ForegroundColor Cyan
$allOk = $true
foreach ($j in $Jobs) {
    if (-not (Test-Path -LiteralPath $j.Dst)) { Write-Host ("  {0,-14} NOT MOVED" -f $j.Name) -ForegroundColor Red; $allOk=$false; continue }
    $d = Get-TreeStats $j.Dst
    $ok = ($d.Files -eq $j.PreFiles) -and ($d.Bytes -eq $j.PreBytes)
    if (-not $ok) { $allOk = $false }
    "{0,-14} before {1,7} files {2,8:N2} GB | after {3,7} files {4,8:N2} GB | {5}" -f `
        $j.Name, $j.PreFiles, ($j.PreBytes/1GB), $d.Files, ($d.Bytes/1GB), $(if($ok){'OK'}else{'MISMATCH'})
}

Write-Host ""
if ($allOk) {
    Write-Host "Move complete and verified." -ForegroundColor Green
    Write-Host "`nNext:" -ForegroundColor Cyan
    Write-Host "  1. Restart OneDrive (Start menu > OneDrive). It will sync the deletion"
    Write-Host "     of the old folders from the cloud -- expected."
    Write-Host "  2. Open C:\dev\Clim_Growth\Paper_1 in VS Code, rebuild the devcontainer,"
    Write-Host "     confirm the files are visible inside the container."
    Write-Host "  3. Re-point the Cowork connected folder to C:\dev\Clim_Growth\Paper_1."
    Write-Host "  4. LOCAL ONLY now -- push to a git remote." -ForegroundColor Yellow
} else {
    Write-Host "Not everything moved. Nothing was deleted; sources that failed are intact." -ForegroundColor Yellow
}
