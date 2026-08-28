<#
    Find what is filling up C: -- REPORT ONLY, deletes nothing.

    Prints, in order:
      1. Drive free/total
      2. Known reclaimable caches, each with its size and the exact command to clear it
      3. Docker / WSL virtual disks (usually the single biggest win on a dev machine)
      4. Largest folders in your user profile
      5. Largest individual files on C:

    Everything is classified SAFE / REVIEW / CAREFUL. Nothing is removed; copy the
    command for whichever item you decide to reclaim.

    Run:
        powershell -ExecutionPolicy Bypass -File .\scripts\2026-08-28-analyse-c-drive.ps1
#>

$ErrorActionPreference = 'SilentlyContinue'
function GB([double]$b) { "{0,8:N2} GB" -f ($b / 1GB) }
function DirSize([string]$p) {
    if (-not (Test-Path -LiteralPath $p)) { return $null }
    (Get-ChildItem -LiteralPath $p -Recurse -File -Force |
        Measure-Object -Property Length -Sum).Sum
}

Write-Host "`n=========== 1. DRIVES ===========" -ForegroundColor Cyan
Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Used -ne $null } | ForEach-Object {
    "  {0}:  used {1}  free {2}  ({3:N0}% free)" -f $_.Name, (GB $_.Used), (GB $_.Free),
        (100 * $_.Free / ($_.Used + $_.Free))
}

Write-Host "`n=========== 2. RECLAIMABLE CACHES ===========" -ForegroundColor Cyan
$targets = @(
  @{ R='SAFE';    N='User temp';             P="$env:TEMP";
     C='Remove-Item "$env:TEMP\*" -Recurse -Force -EA SilentlyContinue' },
  @{ R='SAFE';    N='Windows temp';          P="$env:SystemRoot\Temp";
     C='Remove-Item "$env:SystemRoot\Temp\*" -Recurse -Force -EA SilentlyContinue' },
  @{ R='SAFE';    N='Windows Update cache';  P="$env:SystemRoot\SoftwareDistribution\Download";
     C='net stop wuauserv; Remove-Item "$env:SystemRoot\SoftwareDistribution\Download\*" -Recurse -Force; net start wuauserv' },
  @{ R='SAFE';    N='Recycle Bin';           P='C:\$Recycle.Bin';
     C='Clear-RecycleBin -DriveLetter C -Force' },
  @{ R='SAFE';    N='pip cache';             P="$env:LOCALAPPDATA\pip\Cache";
     C='pip cache purge' },
  @{ R='SAFE';    N='npm cache';             P="$env:APPDATA\npm-cache";
     C='npm cache clean --force' },
  @{ R='SAFE';    N='conda pkgs';            P="$env:USERPROFILE\anaconda3\pkgs";
     C='conda clean --all -y' },
  @{ R='SAFE';    N='NuGet cache';           P="$env:USERPROFILE\.nuget\packages";
     C='dotnet nuget locals all --clear' },
  @{ R='SAFE';    N='VS Code caches';        P="$env:APPDATA\Code\Cache";
     C='Remove-Item "$env:APPDATA\Code\Cache*" -Recurse -Force' },
  @{ R='REVIEW';  N='Downloads';             P="$env:USERPROFILE\Downloads";
     C='# open it and delete what you no longer need' },
  @{ R='REVIEW';  N='Windows.old';           P='C:\Windows.old';
     C='# Settings > System > Storage > Temporary files > Previous Windows installation' },
  @{ R='REVIEW';  N='Hibernation file';      P="$env:SystemRoot\..\hiberfil.sys";
     C='powercfg /h off      # frees ~40% of RAM size; disables hibernate + fast startup' },
  @{ R='CAREFUL'; N='OneDrive local cache';  P="$env:LOCALAPPDATA\Microsoft\OneDrive";
     C='# do not hand-delete; use OneDrive settings > Free up space' }
)
foreach ($t in $targets) {
    $s = DirSize $t.P
    if ($null -eq $s) { continue }
    $col = switch ($t.R) { 'SAFE' {'Green'} 'REVIEW' {'Yellow'} default {'Red'} }
    Write-Host ("  [{0,-7}] {1,-24} {2}" -f $t.R, $t.N, (GB $s)) -ForegroundColor $col
    Write-Host ("            {0}" -f $t.C) -ForegroundColor DarkGray
}

Write-Host "`n=========== 3. DOCKER / WSL VIRTUAL DISKS ===========" -ForegroundColor Cyan
Write-Host "  These grow and never shrink on their own -- often the biggest single win."
$vhdx = Get-ChildItem -Path "$env:LOCALAPPDATA\Docker", "$env:LOCALAPPDATA\Packages",
                            "$env:USERPROFILE\AppData\Local\wsl" -Recurse -Filter *.vhdx -Force |
        Sort-Object Length -Descending
if ($vhdx) {
    $vhdx | Select-Object -First 8 | ForEach-Object { "  {0}  {1}" -f (GB $_.Length), $_.FullName }
    Write-Host "`n  Reclaim inside Docker first (removes unused images/build cache):" -ForegroundColor DarkGray
    Write-Host "      docker system df                 # see what is using space" -ForegroundColor DarkGray
    Write-Host "      docker builder prune -a          # build cache -- usually huge" -ForegroundColor DarkGray
    Write-Host "      docker system prune -a --volumes # CAREFUL: deletes unused images AND volumes" -ForegroundColor DarkGray
    Write-Host "  Then shrink the vhdx itself:" -ForegroundColor DarkGray
    Write-Host '      wsl --shutdown' -ForegroundColor DarkGray
    Write-Host '      Optimize-VHD -Path "<path above>" -Mode Full   # needs Hyper-V module + admin' -ForegroundColor DarkGray
    Write-Host '      # no Hyper-V? use: diskpart > select vdisk file="<path>" > compact vdisk' -ForegroundColor DarkGray
} else {
    Write-Host "  (none found)"
}

Write-Host "`n=========== 4. LARGEST FOLDERS IN YOUR PROFILE ===========" -ForegroundColor Cyan
Get-ChildItem -LiteralPath $env:USERPROFILE -Directory -Force |
    ForEach-Object {
        $s = DirSize $_.FullName
        if ($s) { [pscustomobject]@{ Size = $s; Path = $_.FullName } }
    } | Sort-Object Size -Descending | Select-Object -First 12 |
    ForEach-Object { "  {0}  {1}" -f (GB $_.Size), $_.Path }

Write-Host "`n=========== 5. LARGEST FILES ON C: (>500 MB) ===========" -ForegroundColor Cyan
Get-ChildItem -Path 'C:\' -Recurse -File -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Length -gt 500MB } |
    Sort-Object Length -Descending | Select-Object -First 25 |
    ForEach-Object { "  {0}  {1}" -f (GB $_.Length), $_.FullName }

Write-Host "`nDone. Nothing was deleted." -ForegroundColor Green
