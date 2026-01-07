import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn

# 1. Matches your refactored Train Model (Batch Normalization is key here)
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

# Setup device and Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CircuitNet(num_classes=5).to(device)
model.load_state_dict(torch.load("circuit_model.pth", map_location=device))
model.eval()

class_names = ['bjt', 'capacitor', 'inductor', 'mosfet', 'resistor']
# Mapping for SPICE prefixes
spice_prefixes = {
    'resistor': 'R',
    'capacitor': 'C',
    'inductor': 'L',
    'bjt': 'Q',
    'mosfet': 'M'
}

transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def analyze_circuit(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found!")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Dictionary to keep track of component counts for naming
    counts = {name: 0 for name in class_names}
    found_components = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter noise: typical components are larger than small wire segments
        if w > 15 and h > 15:
            roi = gray[y:y+h, x:x+w]
            roi_pil = Image.fromarray(roi)
            roi_tensor = transform(roi_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(roi_tensor)
                prob = torch.nn.functional.softmax(output[0], dim=0)
                conf, pred = torch.max(prob, 0)
                
            if conf > 0.85:
                label_type = class_names[pred.item()]
                
                # Increment count and create name (e.g., R1, R2)
                counts[label_type] += 1
                unique_name = f"{spice_prefixes[label_type]}{counts[label_type]}"
                
                found_components.append({
                    'name': unique_name,
                    'type': label_type,
                    'bbox': (x, y, w, h)
                })
                
                # Visual Feedback
                color = (0, 255, 0) # Green for components
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                # Display Name and Confidence
                display_text = f"{unique_name} ({conf:.2f})"
                cv2.putText(img, display_text, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    print(f"Analysis Complete. Found: {counts}")
    cv2.imshow("Detected Components with Names", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return found_components

if __name__ == "__main__":
    # Ensure you have a test image ready
    analyze_circuit("D:\codes\ML\image_to_ngspice\dataset\images\circuit_3.png")