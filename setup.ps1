# Sign Language Recognition — Setup Script (Windows PowerShell)
# Run this script AFTER installing Python 3.10+ from https://python.org
#
# Usage:
#   Right-click PowerShell → Run as Administrator (first time only)
#   Then: .\setup.ps1

Write-Host "=== Sign Language Recognition — Project Setup ===" -ForegroundColor Cyan

# 1. Check Python
$pythonExe = $null
foreach ($candidate in @("python", "python3")) {
    try {
        $ver = & $candidate --version 2>&1
        if ($LASTEXITCODE -eq 0) { $pythonExe = $candidate; break }
    } catch {}
}

if (-not $pythonExe) {
    Write-Host "`n[ERROR] Python not found." -ForegroundColor Red
    Write-Host "Download Python 3.10+ from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Make sure to CHECK 'Add Python to PATH' during installation." -ForegroundColor Yellow
    exit 1
}

Write-Host "`n[OK] Found Python: $(& $pythonExe --version)" -ForegroundColor Green

# 2. Create virtual environment
Write-Host "`n[1/4] Creating virtual environment (.venv)..." -ForegroundColor Cyan
& $pythonExe -m venv .venv
if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] Failed to create venv." -ForegroundColor Red; exit 1 }

# 3. Activate & install
Write-Host "[2/4] Installing dependencies..." -ForegroundColor Cyan
& .\.venv\Scripts\pip install --upgrade pip
& .\.venv\Scripts\pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] pip install failed." -ForegroundColor Red; exit 1 }

# 4. Create directory structure
Write-Host "[3/4] Creating project directories..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "data\raw", "data\processed", "models\checkpoints", "results" | Out-Null

# 5. Done
Write-Host "`n[4/4] Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Yellow
Write-Host "  1. Download the dataset from:" -ForegroundColor White
Write-Host "     https://www.kaggle.com/datasets/datamunge/sign-language-mnist" -ForegroundColor Cyan
Write-Host "  2. Place these files in data\raw\:" -ForegroundColor White
Write-Host "       sign_mnist_train.csv" -ForegroundColor Cyan
Write-Host "       sign_mnist_test.csv" -ForegroundColor Cyan
Write-Host ""
Write-Host "  3. Activate the virtual environment:" -ForegroundColor White
Write-Host "       .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "  4. Run a dry-run to validate the pipeline:" -ForegroundColor White
Write-Host "       python src\train.py --dry-run" -ForegroundColor Cyan
Write-Host ""
Write-Host "  5. Start training:" -ForegroundColor White
Write-Host "       python src\train.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "  6. After training, evaluate:" -ForegroundColor White
Write-Host "       python src\evaluate.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "  7. Launch real-time demo:" -ForegroundColor White
Write-Host "       python src\inference.py" -ForegroundColor Cyan
