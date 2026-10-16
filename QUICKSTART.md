# 🚀 Quick Start Guide - Skin Disease AI Detector

This package is completely self-contained. Anyone receiving this folder can run the application immediately **without needing to configure or download anything**.

---

## ⚡ How to Run

### Option 1: Windows (Installation & Run)
1. **To install:** Double-click **`install.bat`** (or right-click `install.ps1` -> "Run with PowerShell").
   - Automatically detects or installs Python 3.11.
   - Detects NVIDIA GPU (CUDA) or CPU automatically.
   - Sets up virtual environment and installs dependencies.
   - Creates a **"Skin Disease AI Detector"** shortcut on your Desktop.
2. **To run anytime:** Double-click **`run.bat`** (or the Desktop shortcut).
   - Starts the server and automatically opens your browser at **`http://127.0.0.1:5000`**!

---

### Option 2: Linux / macOS (One Command)
Open a terminal inside this folder and run:
```bash
./run.sh
```
*(It will set up the virtual environment, launch the server, and open your browser automatically).*

---

### Option 3: Docker (Zero Local Setup)
If you have Docker installed, simply run:
```bash
docker compose up --build
```
Open **`http://127.0.0.1:5000`** in your browser.

---

## 📸 Instant Testing with Demo Samples

You do not need to download the 5GB HAM10000 dataset to test! A curated set of ready-to-test clinical lesion samples is provided inside the **`demo_samples/`** folder:

| Sample Image | Pathology | Risk Category |
| :--- | :--- | :--- |
| `01_Melanoma_Malignant.jpg` | Melanoma | 🔴 Critical Risk |
| `02_Common_Mole_Nevus.jpg` | Melanocytic Nevus (Mole) | 🟢 Low Risk |
| `03_Basal_Cell_Carcinoma.jpg` | Basal Cell Carcinoma | 🟠 High Risk |
| `04_Benign_Keratosis.jpg` | Benign Keratosis | 🟢 Low Risk |
| `05_Actinic_Keratoses.jpg` | Actinic Keratoses | 🟡 Moderate Risk |
| `06_Dermatofibroma.jpg` | Dermatofibroma | 🟢 Low Risk |
| `07_Vascular_Lesion.jpg` | Vascular Lesions | 🟢 Low Risk |

Simply drag & drop any image from `demo_samples/` into the web app to see the live SegFormer boundary delineation mask, clinical ABCDE metrics (area %, asymmetry, border irregularity), and diagnostic differential breakdown!

---

## 📊 Presentation Figures for Faculty

All 300 DPI high-resolution figures for slides and reports are saved in the **`visual_results/`** directory:
* `01_training_metrics_curves.png`: Loss decay & 95%+ accuracy curves
* `02_confusion_matrix_and_roc.png`: 7-class normalized confusion matrix and One-vs-Rest ROC
* `03_visual_segmentation_samples.png`: Multi-sample clinical test panels with overlays & probabilities
* `04_architecture_benchmark_comparison.png`: SegFormer vs UNet vs TransUNet benchmark on RTX 4050
* `training_log.txt`: Official terminal execution log for proof of training
