import os
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.nn.functional as F
from PIL import Image
import io

app = Flask(__name__)

# HAM10000 Dataset actual classes standard
DISEASE_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'mel', 'vasc']
NUM_CLASSES = len(DISEASE_CLASSES)
IMG_SIZE = 224

# Disease details for frontend
DISEASES = {
    'akiec': {
        "name": "Actinic Keratoses (Pre-cancerous)",
        "severity": "Moderate",
        "precaution": "Avoid direct sunlight and use SPF 50+ sunscreen.",
        "icdCode": "L57.0",
        "symptoms": ["Rough patch", "Scaly skin", "Redness"]
    },
    'bcc': {
        "name": "Basal Cell Carcinoma (Cancer)",
        "severity": "High",
        "precaution": "Consult a dermatologist for surgical removal.",
        "icdCode": "C44.91",
        "symptoms": ["Pearly bump", "Pink growth", "Non-healing sore"]
    },
    'bkl': {
        "name": "Benign Keratosis",
        "severity": "Low",
        "precaution": "Generally harmless, but monitor for size changes.",
        "icdCode": "L82.1",
        "symptoms": ["Waxy growth", "Brown/Black color", "Raised surface"]
    },
    'df': {
        "name": "Dermatofibroma",
        "severity": "Low",
        "precaution": "No treatment needed unless it becomes painful.",
        "icdCode": "D21.9",
        "symptoms": ["Small firm bump", "Dimples when pinched", "Itchy"]
    },
    'nv': {
        "name": "Melanocytic Nevi (Common Mole)",
        "severity": "Low",
        "precaution": "Normal mole. Get a checkup if it starts bleeding.",
        "icdCode": "D22.9",
        "symptoms": ["Uniform color", "Smooth edges", "Symmetrical shape"]
    },
    'mel': {
        "name": "Melanoma (Malignant)",
        "severity": "Critical",
        "precaution": "EMERGENCY: Immediate biopsy and specialist consult.",
        "icdCode": "C43.9",
        "symptoms": ["Irregular borders", "Multiple colors", "Rapid growth"]
    },
    'vasc': {
        "name": "Vascular Lesions",
        "severity": "Low",
        "precaution": "Laser therapy available for cosmetic removal.",
        "icdCode": "D18.0",
        "symptoms": ["Red or purple spots", "Small blood vessels visible"]
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading optimized PyTorch AI model... This might take a few seconds.")
try:
    # Rebuild Architecture
    model = models.efficientnet_v2_s()
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(p=0.4),
        nn.Linear(256, NUM_CLASSES)
    )
    
    # Load weights
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode
    print(f"✅ Model loaded successfully on {device}!")
except Exception as e:
    print(f"⚠️ Warning: Failed to load model. Did you run train_model_pytorch.py first? Error: {e}")
    model = None

# Preprocessing match exactly with training
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# HTML Template for Web UI
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Skin Disease AI Tester</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; color: #333; display: flex; flex-direction: column; align-items: center; padding: 40px; }
        .container { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); max-width: 500px; width: 100%; text-align: center; }
        h1 { color: #2c3e50; font-size: 24px; margin-bottom: 20px; }
        input[type="file"] { margin: 20px 0; padding: 10px; border: 1px solid #ddd; border-radius: 6px; width: 90%; }
        button { background-color: #3498db; color: white; border: none; padding: 12px 24px; font-size: 16px; border-radius: 6px; cursor: pointer; transition: background 0.3s; }
        button:hover { background-color: #2980b9; }
        #result { margin-top: 30px; text-align: left; display: none; padding: 15px; border-radius: 8px; background: #e8f4f8; border-left: 5px solid #3498db; }
        #preview { max-width: 100%; max-height: 300px; margin-top: 20px; border-radius: 8px; display: none; }
        .spinner { display: none; margin: 20px auto; border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <h1>🩺 Skin Disease AI Tester</h1>
        <p>Upload a skin lesion image to get an AI prediction.</p>
        
        <input type="file" id="imageInput" accept="image/jpeg, image/png">
        <br>
        <img id="preview" alt="Image Preview">
        <br><br>
        <button onclick="predict()">Analyze Image</button>
        
        <div class="spinner" id="spinner"></div>

        <div id="result">
            <h3 id="diseaseName" style="margin-top:0; color:#2c3e50;"></h3>
            <p><strong>Confidence:</strong> <span id="confidence"></span>%</p>
            <p><strong>Severity:</strong> <span id="severity"></span></p>
            <p><strong>ICD Code:</strong> <span id="icd"></span></p>
            <p><strong>Symptoms:</strong> <span id="symptoms"></span></p>
            <p><strong>Precaution:</strong> <span id="precaution"></span></p>
        </div>
    </div>

    <script>
        const input = document.getElementById('imageInput');
        const preview = document.getElementById('preview');

        input.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                }
                reader.readAsDataURL(file);
            }
        });

        async function predict() {
            const file = input.files[0];
            if (!file) { alert("Please select an image first!"); return; }

            document.getElementById('spinner').style.display = 'block';
            document.getElementById('result').style.display = 'none';

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                
                if (response.ok) {
                    document.getElementById('diseaseName').innerText = data.diseaseName;
                    document.getElementById('confidence').innerText = (data.confidenceScore * 100).toFixed(2);
                    document.getElementById('severity').innerText = data.severity;
                    document.getElementById('icd').innerText = data.icdCode;
                    document.getElementById('symptoms').innerText = data.symptoms.join(', ');
                    document.getElementById('precaution').innerText = data.precaution;
                    document.getElementById('result').style.display = 'block';
                } else {
                    alert("Error: " + data.error);
                }
            } catch (err) {
                alert("Failed to connect to the server.");
            } finally {
                document.getElementById('spinner').style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return INDEX_HTML

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        if model is None:
            return jsonify({"error": "AI Model is not trained or loaded yet. Run train_model_pytorch.py first."}), 500

        file = request.files['file']
        
        # 1. Read and Preprocess Image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device) # Add batch dimension
        
        # 2. Run Inference
        with torch.no_grad():
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)[0]
                
        max_idx = torch.argmax(probabilities).item()
        confidence = probabilities[max_idx].item()
        predicted_class = DISEASE_CLASSES[max_idx]
        
        # 3. Format Response
        result = DISEASES[predicted_class]
        
        response = {
            "diseaseName": result["name"],
            "confidenceScore": round(confidence, 4), # Real AI confidence score!
            "precaution": result["precaution"],
            "severity": result["severity"],
            "icdCode": result["icdCode"],
            "symptoms": result["symptoms"]
        }
        
        print(f"Prediction made: {result['name']} (Confidence: {confidence:.2%})")
        return jsonify(response)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("--- AI Skin Disease Model Server Started ---")
    print("Waiting for requests from Spring Boot at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)