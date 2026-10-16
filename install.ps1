<#
.SYNOPSIS
    Automated installer script for Skin Disease AI Detector on Windows PowerShell.
.DESCRIPTION
    Checks for Python (or installs via winget), sets up .venv, installs PyTorch (CUDA or CPU),
    installs requirements, creates a desktop shortcut, and tests the installation.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  🩺 Skin Disease AI Detector - Windows Setup & Installer (PowerShell)" -ForegroundColor Cyan
Write-Host "  Architecture: SegFormer Vision Transformer (HAM10000 95.42% Accuracy)" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Detect or Install Python
Write-Host "[1/5] Checking for Python installation..." -ForegroundColor Yellow
$pyCmd = $null

if (Get-Command "python" -ErrorAction SilentlyContinue) {
    try {
        $major = & python -c "import sys; print(sys.version_info.major)" 2>$null
        if ($major -eq "3") { $pyCmd = "python" }
    } catch {}
}

if (-not $pyCmd -and (Get-Command "py" -ErrorAction SilentlyContinue)) {
    try {
        $major = & py -3 -c "import sys; print(sys.version_info.major)" 2>$null
        if ($major -eq "3") { $pyCmd = "py" }
    } catch {}
}

if (-not $pyCmd) {
    Write-Host "[WARN] Python 3 not detected in PATH. Attempting automated installation..." -ForegroundColor Yellow
    if (Get-Command "winget" -ErrorAction SilentlyContinue) {
        Write-Host "[INFO] Installing Python 3.11 using winget..." -ForegroundColor Cyan
        & winget install --id Python.Python.3.11 --exact --accept-package-agreements --accept-source-agreements --silent
    } else {
        $tempExe = Join-Path $env:TEMP "python-3.11.9-amd64.exe"
        Write-Host "[INFO] Downloading Python 3.11 from python.org..." -ForegroundColor Cyan
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        (New-Object System.Net.WebClient).DownloadFile("https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe", $tempExe)
        Write-Host "[INFO] Installing Python silently..." -ForegroundColor Cyan
        Start-Process -FilePath $tempExe -ArgumentList "/passive InstallAllUsers=0 PrependPath=1" -Wait
        Remove-Item -Force $tempExe -ErrorAction SilentlyContinue
    }

    # Refresh current environment PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    if (Get-Command "python" -ErrorAction SilentlyContinue) {
        $pyCmd = "python"
    } elseif (Get-Command "py" -ErrorAction SilentlyContinue) {
        $pyCmd = "py"
    } else {
        Write-Error "Python could not be located. Please install Python 3.10+ from python.org with 'Add Python to PATH' enabled."
    }
}

$pyVer = & $pyCmd --version
Write-Host "[OK] Using Python: $pyVer" -ForegroundColor Green

# 2. Virtual Environment
Write-Host ""
Write-Host "[2/5] Setting up virtual environment (.venv)..." -ForegroundColor Yellow
$venvDir = Join-Path $ScriptDir ".venv"
$venvPy = Join-Path $venvDir "Scripts\python.exe"
$venvPip = Join-Path $venvDir "Scripts\pip.exe"

if (-not (Test-Path $venvPy)) {
    Write-Host "[INFO] Creating .venv..." -ForegroundColor Cyan
    if ($pyCmd -eq "py") {
        & py -3 -m venv $venvDir
    } else {
        & python -m venv $venvDir
    }
    Write-Host "[OK] Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "[OK] Existing virtual environment found." -ForegroundColor Green
}

# 3. Upgrade Pip
Write-Host ""
Write-Host "[3/5] Upgrading pip and packaging utilities..." -ForegroundColor Yellow
& $venvPy -m pip install --upgrade pip setuptools wheel --quiet

# 4. Install PyTorch & Dependencies
Write-Host ""
Write-Host "[4/5] Installing project dependencies (PyTorch, Transformers, Flask)..." -ForegroundColor Yellow
$hasGpu = $false
try {
    if (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue) {
        & nvidia-smi | Out-Null
        $hasGpu = $true
    }
} catch {}

if ($hasGpu) {
    Write-Host "[INFO] NVIDIA GPU detected. Installing CUDA 12.1 PyTorch build..." -ForegroundColor Cyan
    & $venvPip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
} else {
    Write-Host "[INFO] Installing PyTorch CPU build..." -ForegroundColor Cyan
    & $venvPip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
}

Write-Host "[INFO] Installing requirements.txt..." -ForegroundColor Cyan
& $venvPip install -r (Join-Path $ScriptDir "requirements.txt") --quiet

# 5. Verification & Shortcut
Write-Host ""
Write-Host "[5/5] Verifying installation..." -ForegroundColor Yellow
& $venvPy -c "import torch, torchvision, transformers, flask, PIL, numpy; print(f'[OK] Modules loaded. PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')"

$modelPath = Join-Path $ScriptDir "best_segformer_model.pth"
if (Test-Path $modelPath) {
    Write-Host "[OK] Model weights found: best_segformer_model.pth" -ForegroundColor Green
} else {
    Write-Host "[WARN] Model file 'best_segformer_model.pth' not found in project directory." -ForegroundColor Yellow
}

# Create Desktop Shortcut
try {
    $ws = New-Object -ComObject WScript.Shell
    $desktop = [System.Environment]::GetFolderPath("Desktop")
    $shortcutPath = Join-Path $desktop "Skin Disease AI Detector.lnk"
    $shortcut = $ws.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = Join-Path $ScriptDir "run.bat"
    $shortcut.WorkingDirectory = $ScriptDir
    $shortcut.Description = "Launch Skin Disease AI Detector (SegFormer)"
    $shortcut.Save()
    Write-Host "[OK] Created desktop shortcut: Skin Disease AI Detector" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Could not create desktop shortcut automatically." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Green
Write-Host "  🎉 Installation Completed Successfully!" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Green
Write-Host "To run the application:"
Write-Host "  - Double-click the desktop shortcut 'Skin Disease AI Detector'"
Write-Host "  - Or execute .\run.bat from this directory"
Write-Host ""
