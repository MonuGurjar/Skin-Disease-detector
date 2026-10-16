#!/usr/bin/env bash
# ==============================================================================
# 🩺 Skin Disease AI Detector - Automated Launcher (Linux & macOS)
# Double-click or run: ./run.sh
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "  🩺 Starting Skin Disease AI Detector (SegFormer Vision Transformer)"
echo "========================================================================"

# 1. Check Python
if command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
else
    echo "❌ Error: Python 3 is not installed on this system."
    echo "Please install Python 3 (https://www.python.org/downloads/) and re-run."
    exit 1
fi

echo "✔ Detected Python: $($PY_CMD --version)"

# 2. Setup Virtual Environment if not present
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment (.venv)..."
    if command -v uv &>/dev/null; then
        uv venv .venv
    else
        $PY_CMD -m venv .venv
    fi
fi

# 3. Activate venv
source .venv/bin/activate

# 4. Check & Install Dependencies
if ! python -c "import flask, torch, transformers, PIL" &>/dev/null; then
    echo "📥 Installing required dependencies (PyTorch, Transformers, Flask)..."
    echo "This may take 1-2 minutes on first launch..."
    if command -v uv &>/dev/null; then
        uv pip install -r requirements.txt
    else
        pip install --upgrade pip
        pip install -r requirements.txt
    fi
fi

echo "✔ All dependencies ready!"
echo "🚀 Launching AI Server at http://127.0.0.1:5000 ..."
echo ""
echo "👉 You can test with sample images in the 'demo_samples/' folder!"
echo ""

# 5. Open browser in background after 2 seconds
(
    sleep 2
    if command -v xdg-open &>/dev/null; then
        xdg-open "http://127.0.0.1:5000" &>/dev/null || true
    elif command -v open &>/dev/null; then
        open "http://127.0.0.1:5000" &>/dev/null || true
    fi
) &

# 6. Run Flask Application
exec python app.py
