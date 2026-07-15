$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPath = Join-Path $ProjectRoot ".venv"
Set-Location -LiteralPath $ProjectRoot

function Remove-SafeCachePath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $Resolved = (Resolve-Path -LiteralPath $Path).Path
    if (-not $Resolved.StartsWith($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to delete a path outside the project: $Resolved"
    }
    Remove-Item -LiteralPath $Resolved -Recurse -Force
    Write-Host "Removed cache: $Resolved"
}

Remove-SafeCachePath (Join-Path $ProjectRoot ".pytest_cache")
Remove-SafeCachePath (Join-Path $ProjectRoot ".ruff_cache")

$CacheDirectories = Get-ChildItem -LiteralPath $ProjectRoot -Directory -Filter "__pycache__" -Recurse
foreach ($Directory in $CacheDirectories) {
    if (-not $Directory.FullName.StartsWith($VenvPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        Remove-SafeCachePath $Directory.FullName
    }
}

Get-ChildItem -LiteralPath $ProjectRoot -File -Filter "*.pyc" -Recurse | ForEach-Object {
    $Resolved = $_.FullName
    if (-not $Resolved.StartsWith($VenvPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        if (-not $Resolved.StartsWith($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to delete a path outside the project: $Resolved"
        }
        Remove-Item -LiteralPath $Resolved -Force
        Write-Host "Removed cache: $Resolved"
    }
}

Write-Host "Cache cleanup completed. .env, .venv, and all *.db files were preserved."
