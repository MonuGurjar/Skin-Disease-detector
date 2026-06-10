#!/bin/bash
echo "=========================================="
echo "🩺 Installing Skin Disease AI Detector"
echo "=========================================="

# Check if uv is installed, if not, install it
if ! command -v uv &> /dev/null; then
    echo "Installing 'uv' (Fast Python Package Manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Clone the repo
if [ ! -d "Skin-Disease-detector" ]; then
    echo "Cloning repository..."
    git clone https://github.com/MonuGurjar/Skin-Disease-detector.git
fi

cd Skin-Disease-detector

echo "Setting up Virtual Environment..."
uv venv

# Activate venv
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

echo "Installing Dependencies (PyTorch & Libraries)..."
uv pip install -r requirements.txt

echo "=========================================="
echo "✅ Installation Complete!"
echo ""
echo "To start the web application, run:"
echo "cd Skin-Disease-detector"
echo "source .venv/bin/activate"
echo "python app.py"
echo "=========================================="
