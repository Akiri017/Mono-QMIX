param(
    [string]$RepoUrl = "https://github.com/oxwhirl/pymarl.git",
    [string]$TargetDir = "pymarl"
)

$ErrorActionPreference = "Stop"

if (Test-Path $TargetDir) {
    Write-Host "Target already exists: $TargetDir"
    exit 0
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "git not found on PATH. Install Git for Windows and retry."
}

git clone $RepoUrl $TargetDir
Write-Host "Cloned PyMARL into $TargetDir"
