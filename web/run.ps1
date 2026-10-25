# ============================================================================
#  SmartCross — one command to get the console running
#
#      .\run.ps1              build the model if needed, then serve
#      .\run.ps1 -Rebuild     force a retrain first
#
#  If PowerShell refuses to run this file, allow local scripts once:
#      Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# ============================================================================

param([switch]$Rebuild)

$ErrorActionPreference = "Stop"
$here    = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $here "backend"
$model   = Join-Path $backend "model\smartcross.pkl"

Write-Host ""
Write-Host "======================================================================"
Write-Host " SMARTCROSS" -ForegroundColor Yellow
Write-Host "======================================================================"

# ---- find a python
$py = $null
foreach ($c in @("python", "py", "python3")) {
    try { if (& $c --version 2>$null) { $py = $c; break } } catch {}
}
if (-not $py) {
    Write-Host "`n[X] No python on PATH. Install Python 3.10+ and reopen the terminal.`n" -ForegroundColor Red
    exit 1
}
Write-Host "`n  python   : $(& $py --version 2>&1)"

# ---- packages
$needed = @("fastapi", "uvicorn", "pydantic", "sklearn", "joblib", "numpy")
$missing = @()
foreach ($m in $needed) {
    & $py -c "import $m" 2>$null
    if ($LASTEXITCODE -ne 0) { $missing += $m }
}
if ($missing.Count -gt 0) {
    $pipNames = $missing -replace "^sklearn$", "scikit-learn"
    Write-Host "  installing: $($pipNames -join ', ')" -ForegroundColor Yellow
    & $py -m pip install --quiet --disable-pip-version-check @pipNames
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[X] pip install failed. Run it yourself:" -ForegroundColor Red
        Write-Host "    $py -m pip install $($pipNames -join ' ')`n"
        exit 1
    }
} else {
    Write-Host "  packages : all present"
}

# ---- model
# Existence is not enough: a pickle written by different numpy/sklearn versions
# is present but unreadable. Actually open it before trusting it.
$needBuild = $Rebuild -or -not (Test-Path $model)
$reason = if ($Rebuild) { "forced rebuild" } else { "no model yet" }

if (-not $needBuild) {
    Push-Location $backend
    # also refuse a model from before the fused hybrid: it opens, but has no peer tables
    & $py -c "import joblib,sys; sys.exit(0 if 'hist_names' in joblib.load(r'$model') else 3)" 2>$null
    $loadOk = ($LASTEXITCODE -eq 0)
    Pop-Location
    if (-not $loadOk) {
        $needBuild = $true
        $reason = "the existing model is unreadable here or predates the fused hybrid"
        Write-Host "`n  [!] smartcross.pkl cannot be read by this Python." -ForegroundColor Yellow
        Write-Host "      Rebuilding it with your own numpy/scikit-learn." -ForegroundColor Yellow
    } else {
        $mb = [math]::Round((Get-Item $model).Length / 1MB, 1)
        Write-Host "  model    : smartcross.pkl, $mb MB, built $((Get-Item $model).LastWriteTime)"
    }
}

if ($needBuild) {
    Write-Host "`n  building the model ($reason)." -ForegroundColor Yellow
    Write-Host "  This takes a few minutes. Leave the window open.`n" -ForegroundColor Yellow
    & $py -c "import xgboost" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  note: xgboost is not installed, so that row will be absent from" -ForegroundColor DarkGray
        Write-Host "        the engine table. 'pip install xgboost' to measure it too.`n" -ForegroundColor DarkGray
    }
    Push-Location $backend
    try {
        & $py build_model.py
        if ($LASTEXITCODE -ne 0) { throw "build_model.py failed" }
    } finally { Pop-Location }
}

# ---- serve
Write-Host "`n======================================================================"
Write-Host "  Opening http://127.0.0.1:8000   (Ctrl+C to stop)" -ForegroundColor Green
Write-Host "======================================================================`n"

Start-Job -ScriptBlock {
    Start-Sleep -Seconds 3
    Start-Process "http://127.0.0.1:8000"
} | Out-Null

Push-Location $backend
try { & $py app.py } finally { Pop-Location }
