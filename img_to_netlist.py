import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn

# --- 1. Model Architecture ---
class CircuitNet(nn.Module):
    def __init__(self, num_classes=7): # UPDATED: 7 Classes now
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

# --- 2. Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ALPHABETICAL ORDER is critical here. Check your dataset_v3 folder names!
class_names = ['bjt', 'capacitor', 'inductor', 'mosfet', 'resistor', 'voltage', 'wire']

spice_prefixes = {
    'resistor': 'R', 'capacitor': 'C', 'inductor': 'L', 
    'bjt': 'Q', 'mosfet': 'M', 'voltage': 'V'
}

model = CircuitNet(num_classes=len(class_names)).to(device)
try:
    model.load_state_dict(torch.load("circuit_model.pth", map_location=device))
    model.eval()
except:
    print("Please retrain the model on dataset_v3 first!")
    exit()

transform = transforms.Compose([
    transforms.Grayscale(1), transforms.Resize((64, 64)),
    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
])

# --- 3. Main Logic ---
def generate_netlist(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Invert for contour finding (White component on Black background)
    contours, _ = cv2.findContours(255 - thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    components = []
    counts = {name: 0 for name in class_names}

    print("--- Detecting Components ---")
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter tiny noise
        if w > 15 and h > 15:
            # ROI for prediction
            roi = gray[y:y+h, x:x+w]
            roi_pil = Image.fromarray(roi)
            roi_tensor = transform(roi_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(roi_tensor)
                prob = torch.nn.functional.softmax(output[0], dim=0)
                conf, pred = torch.max(prob, 0)
            
            label = class_names[pred.item()]
            
            # --- THE FIX IS HERE ---
            if label == 'wire':
                # The AI says this is a wire, so we ignore it as a component
                continue 
            
            if conf > 0.7:
                counts[label] += 1
                name = f"{spice_prefixes[label]}{counts[label]}"
                
                print(f"Found {name} ({label}) - Conf: {conf:.2f}")
                
                components.append({
                    'name': name, 'type': label, 'box': (x, y, w, h)
                    # Add pin detection/connectivity logic here...
                })
                
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    generate_netlist("dataset/images/circuit_2.png")