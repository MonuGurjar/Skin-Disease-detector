@echo off
setlocal
title Skin Disease AI Detector
cd /d "%~dp0"

echo ========================================================================
echo   🩺 Starting Skin Disease AI Detector (SegFormer Vision Transformer)
echo ========================================================================
echo.

:: 1. Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    where py >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Python is not installed or not in your PATH.
        echo Please install Python 3.10+ from https://www.python.org/downloads/
        echo (Make sure to check "Add Python to PATH" during installation)
        pause
        exit /b 1
    ) else (
        set "PY=py"
    )
) else (
    set "PY=python"
)

echo [OK] Python detected.

:: 2. Create Virtual Environment if not present
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment (.venv)...
    %PY% -m venv .venv
)

:: 3. Activate Virtual Environment
call .venv\Scripts\activate.bat

:: 4. Check & Install Dependencies
python -c "import flask, torch, transformers, PIL" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing required dependencies (PyTorch, Transformers, Flask)...
    echo This may take 1-2 minutes on first launch...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
)

echo [OK] Environment ready!
echo [INFO] Starting web application at http://127.0.0.1:5000 ...
echo [INFO] Sample test images are available in the "demo_samples" folder!
echo.

:: 5. Open Browser after 3 seconds
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:5000"

:: 6. Run the Flask Web Application
python app.py

pause
