# 🩺 Skin Disease AI Detector

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Flask](https://img.shields.io/badge/Flask-API-black.svg)

An advanced machine learning system designed to recognize and diagnose skin diseases. It utilizes a fine-tuned **EfficientNetV2** model written in **PyTorch** and deployed via a **Flask API**. The system is highly optimized for fast inference and training, utilizing Mixed Precision (AMP) and modern hardware acceleration.

This project can accurately classify 7 different types of skin lesions based on the HAM10000 dataset:
1. Actinic Keratoses (Pre-cancerous)
2. Basal Cell Carcinoma (Cancer)
3. Benign Keratosis
4. Dermatofibroma
5. Melanocytic Nevi (Common Mole)
6. Melanoma (Malignant)
7. Vascular Lesions

---

## 🚀 Quick Install (One-Command)

To instantly install this project on any Linux/macOS machine, run this simple curl command in your terminal. It will automatically download the code, set up a virtual environment using `uv`, and install all PyTorch dependencies:

```bash
curl -fsSL https://raw.githubusercontent.com/MonuGurjar/Skin-Disease-detector/main/install.sh | bash
```

Once the installation finishes, simply start the web app:
```bash
cd Skin-Disease-detector
source .venv/bin/activate
python app.py
```

---

## 🛠️ Manual Setup Guide

If you prefer to set up the project manually or are on Windows, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/MonuGurjar/Skin-Disease-detector.git
cd Skin-Disease-detector
```

### 2. Create a Virtual Environment
We recommend using [uv](https://github.com/astral-sh/uv) for lightning-fast package management, but standard `venv` works too.
```bash
uv venv
```

### 3. Activate the Environment
* **Linux/macOS:** `source .venv/bin/activate`
* **Windows:** `.venv\Scripts\activate`

### 4. Install Dependencies
```bash
uv pip install -r requirements.txt
```

---

## 🧠 Training the Model

If you wish to train the model yourself (or fine-tune it further), ensure your dataset images are located inside the `dataset_dir/` folder. 

Then run the PyTorch training script:
```bash
python train_model_pytorch.py
```
* **Phase 1 (Transfer Learning):** Freezes the base EfficientNetV2 model and trains only the classifier head rapidly.
* **Phase 2 (Fine-tuning):** Unfreezes the base network to deeply learn skin features, utilizing PyTorch `ReduceLROnPlateau` for optimal convergence.

The final weights will be saved as `model.pth`.

---

## 💻 Running the Web Application

To use the AI via an interactive web interface:

```bash
python app.py
```

1. Open your browser and navigate to `http://127.0.0.1:5000`
2. You will see the **Skin Disease AI Tester** interface.
3. Upload any `.jpg` skin lesion image.
4. The AI will instantly return the diagnosis, confidence score, severity, ICD-10 Code, and recommended precautions!

### Quick Terminal Testing
If you want to test a single image purely from the command line without the web server:
```bash
python test_inference_pytorch.py dataset_dir/HAM10000_images_part_1/ISIC_0027419.jpg
```

---
<div align="center">
  Built with ❤️ for Medical AI Research
</div>
