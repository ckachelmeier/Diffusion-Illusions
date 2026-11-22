# Copies the results (the untracked folder: runs and logs) from a vast.ai instance into this project's untracked folder.
#     .\vast\pull_results.ps1 -Address 123.45.67.89 -Port 41234
# Address and port come from the instance's Connect button in vast.ai, which shows: ssh -p PORT root@ADDRESS
# Files that exist here too are overwritten by the instance's copy. Run folder names are unique, so new runs never collide.
param(
    [Parameter(Mandatory = $true)][string]$Address,
    [Parameter(Mandatory = $true)][int]$Port,
    [string]$User = 'root',
    [string]$RemoteDir = '/workspace/diffusion-illusion'
)

$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$target = "$User@$Address"
$remoteArchive = '/tmp/untracked-results.tgz'
$localArchive = Join-Path $env:TEMP 'vast-untracked-results.tgz'
$tar = Join-Path $env:SystemRoot 'System32\tar.exe'  #Windows' own tar, not Git's, which misreads C:\ paths

function Invoke-Checked([string]$what, [scriptblock]$command) {
    Write-Host "==> $what"
    & $command
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

Invoke-Checked 'Packing results on the instance' {
    ssh -p $Port $target "cd '$RemoteDir' && tar czf $remoteArchive untracked && du -h $remoteArchive"
}
Invoke-Checked 'Downloading' {
    scp -P $Port "${target}:$remoteArchive" $localArchive
}
Invoke-Checked "Unpacking into $projectDir\untracked" {
    & $tar -xzf $localArchive -C $projectDir
}
Remove-Item $localArchive
ssh -p $Port $target "rm -f $remoteArchive" | Out-Null
Write-Host 'Done.'
