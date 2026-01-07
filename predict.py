import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# --- 1. The Architecture (Must match train_model.py exactly) ---
class CircuitNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# --- 2. Predictor Class ---
class ComponentPredictor:
    def __init__(self, model_path="circuit_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['bjt', 'capacitor', 'inductor', 'mosfet', 'resistor']
        
        # Initialize and load model
        self.model = CircuitNet(num_classes=len(self.class_names)).to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded successfully from {model_path}")
        else:
            print("Error: .pth file not found! Train the model first.")

        # Must match training normalization
        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def predict(self, image_path):
        """Takes an image path and returns (label, confidence_score)"""
        try:
            img = Image.open(image_path).convert('L')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
            
            label = self.class_names[predicted_idx.item()]
            score = confidence.item()
            return label, score
        except Exception as e:
            return f"Error: {e}", 0.0

# --- 3. Standalone Test ---
if __name__ == "__main__":
    predictor = ComponentPredictor("circuit_model.pth")
    
    # Test on a sample from your dataset
    test_path = "dataset_v2/resistor/resistor_0_noisy.png" # Adjust path as needed
    
    if os.path.exists(test_path):
        label, conf = predictor.predict(test_path)
        print(f"\nTarget Image: {test_path}")
        print(f"Result: {label.upper()} ({conf*100:.2f}% confidence)")
    else:
        print(f"Please provide a valid test image at {test_path}")