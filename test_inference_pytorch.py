import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.nn.functional as F
from PIL import Image

DISEASE_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'mel', 'vasc']
NUM_CLASSES = len(DISEASE_CLASSES)
IMG_SIZE = 224

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_inference_pytorch.py <path_to_image.jpg>")
        print("Example: python test_inference_pytorch.py dataset_dir/HAM10000_images_part_1/ISIC_0027419.jpg")
        return

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"❌ Error: File not found -> {img_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading optimized PyTorch AI model onto {device}...")
    
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
    except Exception as e:
        print(f"❌ Error: Failed to load model. Did you run train_model_pytorch.py first? ({e})")
        return

    # Preprocessing match exactly with training
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Processing image: {img_path}")
    try:
        img = Image.open(img_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return

    print("Running inference...")
    with torch.no_grad():
        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)[0]
    
    # Sort predictions by confidence
    results = [(DISEASE_CLASSES[i], probabilities[i].item() * 100.0) for i in range(len(probabilities))]
    results.sort(reverse=True, key=lambda x: x[1])

    print("\n" + "="*40)
    print("🎯 PREDICTION RESULTS")
    print("="*40)
    for i, (class_name, prob) in enumerate(results):
        if i == 0:
            print(f"🏆 Top Match: {class_name:5s} -> {prob:6.2f}% confidence")
            print("-" * 40)
        else:
            print(f"   Option {i+1}: {class_name:5s} -> {prob:6.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()
