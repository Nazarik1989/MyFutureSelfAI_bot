$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $ProjectRoot

function Assert-NativeSuccess {
    param([Parameter(Mandatory = $true)][string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

$VenvPath = Join-Path $ProjectRoot ".venv"
$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$BotCommand = Join-Path $VenvPath "Scripts\future-self-bot.exe"
$EnvPath = Join-Path $ProjectRoot ".env"

if (-not (Test-Path -LiteralPath $ActivateScript)) {
    Write-Error ".venv is missing. Run .\scripts\setup_windows.ps1 first."
}
if (-not (Test-Path -LiteralPath $EnvPath)) {
    Write-Error ".env is missing. Run .\scripts\setup_windows.ps1 first."
}

. $ActivateScript
Write-Host "Applying database migrations..."
& $VenvPython -m alembic upgrade head
Assert-NativeSuccess "Database migration"

Write-Host "Starting the Telegram bot. Press Ctrl+C to stop."
if (-not (Test-Path -LiteralPath $BotCommand)) {
    Write-Error "future-self-bot is not installed. Run setup_windows.ps1 again."
}
& $BotCommand
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
