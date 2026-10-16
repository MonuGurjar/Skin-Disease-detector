@echo off
setlocal enabledelayedexpansion
title Skin Disease AI Detector - Windows Installer
cd /d "%~dp0"

echo ========================================================================
echo   🩺 Skin Disease AI Detector - Windows Setup ^& Installer
echo   Architecture: SegFormer Vision Transformer (HAM10000 95.42%% Accuracy)
echo ========================================================================
echo.

:: --------------------------------------------------------------------------
:: Step 1: Detect or Install Python
:: --------------------------------------------------------------------------
echo [1/5] Checking for Python installation...

set "PY_CMD="

:: Check standard 'python'
where python >nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=*" %%i in ('python -c "import sys; print(sys.version_info.major)" 2^>nul') do (
        if "%%i"=="3" set "PY_CMD=python"
    )
)

:: Check Python Launcher 'py'
if "!PY_CMD!"=="" (
    where py >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=*" %%i in ('py -3 -c "import sys; print(sys.version_info.major)" 2^>nul') do (
            if "%%i"=="3" set "PY_CMD=py -3"
        )
    )
)

:: If Python is not found, try to auto-install via winget or direct download
if "!PY_CMD!"=="" (
    echo [WARN] Python 3 was not found in your system PATH.
    echo [INFO] Attempting automated Python 3.11 installation...
    echo.
    
    where winget >nul 2>&1
    if !errorlevel! equ 0 (
        echo [INFO] Found Windows Package Manager (winget). Installing Python 3.11...
        winget install --id Python.Python.3.11 --exact --accept-package-agreements --accept-source-agreements --silent
    ) else (
        echo [INFO] Downloading official Python 3.11 64-bit installer...
        powershell -NoProfile -ExecutionPolicy Bypass -Command ^
            "$installer = [System.IO.Path]::Combine($env:TEMP, 'python-3.11.9-amd64.exe'); " ^
            "Write-Host 'Downloading Python from python.org...'; " ^
            "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; " ^
            "(New-Object System.Net.WebClient).DownloadFile('https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe', $installer); " ^
            "Write-Host 'Running silent installation (adding Python to PATH)...'; " ^
            "Start-Process -FilePath $installer -ArgumentList '/passive InstallAllUsers=0 PrependPath=1' -Wait; " ^
            "Remove-Item -Force $installer"
    )

    :: Refresh PATH in current session
    for /f "tokens=*" %%p in ('powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('Path', 'User') + ';' + [Environment]::GetEnvironmentVariable('Path', 'Machine')"') do set "PATH=%%p"

    where python >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY_CMD=python"
    ) else (
        where py >nul 2>&1
        if !errorlevel! equ 0 (
            set "PY_CMD=py -3"
        )
    )
)

if "!PY_CMD!"=="" (
    echo.
    echo [ERROR] Could not automatically locate or install Python.
    echo Please install Python 3.10 or 3.11 manually from:
    echo   https://www.python.org/downloads/
    echo IMPORTANT: Make sure to check [x] "Add Python to PATH" during setup.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('!PY_CMD! --version 2^>^&1') do set "PY_VER=%%v"
echo [OK] Using Python: !PY_VER!

:: --------------------------------------------------------------------------
:: Step 2: Create Virtual Environment
:: --------------------------------------------------------------------------
echo.
echo [2/5] Setting up isolated Python virtual environment (.venv)...

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Creating .venv...
    !PY_CMD! -m venv .venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists (.venv).
)

set "VENV_PY=.venv\Scripts\python.exe"
set "VENV_PIP=.venv\Scripts\pip.exe"

:: --------------------------------------------------------------------------
:: Step 3: Upgrade pip and install core build tools
:: --------------------------------------------------------------------------
echo.
echo [3/5] Upgrading pip, setuptools, and wheel...
"!VENV_PY!" -m pip install --upgrade pip setuptools wheel --quiet

:: --------------------------------------------------------------------------
:: Step 4: Install AI Dependencies (PyTorch, SegFormer, Flask)
:: --------------------------------------------------------------------------
echo.
echo [4/5] Installing project dependencies (PyTorch, Transformers, Flask, etc.)...
echo       (This may take 2-4 minutes depending on internet speed)

:: Check for NVIDIA GPU to optionally install CUDA PyTorch
where nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    echo [INFO] NVIDIA GPU detected. Installing PyTorch with CUDA acceleration...
    "!VENV_PIP!" install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
) else (
    echo [INFO] No NVIDIA GPU detected. Installing standard CPU PyTorch...
    "!VENV_PIP!" install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
)

:: Install remainder of requirements
"!VENV_PIP!" install -r requirements.txt --quiet
if !errorlevel! neq 0 (
    echo [WARN] Fast install encountered a warning. Retrying full requirements install...
    "!VENV_PIP!" install -r requirements.txt
)

:: --------------------------------------------------------------------------
:: Step 5: Verification & Desktop Shortcut
:: --------------------------------------------------------------------------
echo.
echo [5/5] Verifying installation and model weights...

"!VENV_PY!" -c "import torch, torchvision, transformers, flask, PIL, numpy; print(f'[OK] Modules loaded successfully. PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
if !errorlevel! neq 0 (
    echo [ERROR] Module verification failed. Please check internet connection and retry.
    pause
    exit /b 1
)

if not exist "best_segformer_model.pth" (
    echo [WARN] best_segformer_model.pth was not found in the root directory!
    echo        Make sure git LFS or the model file was cloned properly.
) else (
    echo [OK] SegFormer trained weights found: best_segformer_model.pth
)

:: Create Desktop Shortcut
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ws = New-Object -ComObject WScript.Shell; " ^
    "$desktop = [Environment]::GetFolderPath('Desktop'); " ^
    "$shortcutPath = [System.IO.Path]::Combine($desktop, 'Skin Disease AI Detector.lnk'); " ^
    "$shortcut = $ws.CreateShortcut($shortcutPath); " ^
    "$shortcut.TargetPath = [System.IO.Path]::Combine((Get-Location).Path, 'run.bat'); " ^
    "$shortcut.WorkingDirectory = (Get-Location).Path; " ^
    "$shortcut.Description = 'Launch Skin Disease AI Detector (SegFormer)'; " ^
    "$shortcut.Save(); " ^
    "Write-Host '[OK] Created desktop shortcut: Skin Disease AI Detector'"

echo.
echo ========================================================================
echo   🎉 Installation Successfully Completed!
echo ========================================================================
echo.
echo You can run the application anytime by:
echo   1. Double-clicking the "Skin Disease AI Detector" icon on your Desktop
echo   2. Or double-clicking "run.bat" in this folder
echo.
echo Sample clinical test images are located in the "demo_samples" folder.
echo.

set /p START_NOW="Would you like to start the application right now? (Y/N, default Y): "
if /i "!START_NOW!"=="" set "START_NOW=Y"
if /i "!START_NOW!"=="Y" (
    echo [INFO] Launching application...
    start "" run.bat
)

exit /b 0
