$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $ProjectRoot

function Assert-NativeSuccess {
    param([Parameter(Mandatory = $true)][string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

$PythonCommand = $null
$PythonArgs = @()
if (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonCommand = "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonCommand = "py"
    $PythonArgs = @("-3")
} else {
    Write-Error "Python was not found. Install Python 3.12+ from python.org and retry."
}

$VersionText = & $PythonCommand @PythonArgs -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
Assert-NativeSuccess "Python version check"
$PythonVersion = [version]$VersionText.Trim()
if ($PythonVersion -lt [version]"3.12") {
    Write-Error "Python 3.12 or newer is required. Found $PythonVersion."
}
Write-Host "Found Python $PythonVersion."

$VenvPath = Join-Path $ProjectRoot ".venv"
if (-not (Test-Path -LiteralPath $VenvPath)) {
    Write-Host "Creating .venv..."
    & $PythonCommand @PythonArgs -m venv $VenvPath
    Assert-NativeSuccess "Virtual environment creation"
} else {
    Write-Host "Using existing .venv."
}

$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $ActivateScript)) {
    Write-Error "Activation script is missing. Remove the broken .venv manually and retry."
}
. $ActivateScript

Write-Host "Installing the project and development dependencies..."
& $VenvPython -m pip install --upgrade pip
Assert-NativeSuccess "pip upgrade"
& $VenvPython -m pip install -e ".[dev]"
Assert-NativeSuccess "Project installation"

$EnvPath = Join-Path $ProjectRoot ".env"
if (-not (Test-Path -LiteralPath $EnvPath)) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot ".env.example") -Destination $EnvPath
    Write-Host "Created .env from .env.example."
} else {
    Write-Host "Kept the existing .env unchanged."
}

Write-Host "Applying database migrations..."
& $VenvPython -m alembic upgrade head
Assert-NativeSuccess "Database migration"

Write-Host ""
Write-Host "Setup completed."
Write-Host "1. Fill TELEGRAM_BOT_TOKEN and AI_API_KEY in .env."
Write-Host "2. Run: .\.venv\Scripts\python.exe -m future_self.doctor"
Write-Host "3. Start the bot: .\scripts\run_windows.ps1"
Write-Host "The bot was not started automatically."
