import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from modules.config import cfg  # <--- IMPORT CONFIG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CircuitNet(nn.Module):
    def __init__(self, num_classes=None): 
        super().__init__()
        # Use config for dynamic class count
        n_classes = num_classes if num_classes else len(cfg.class_names)
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, n_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

class ComponentDetector:
    def __init__(self, model_path):
        self.model = CircuitNet(num_classes=len(cfg.class_names)).to(DEVICE)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            self.model.eval()
        except FileNotFoundError:
            print(f"❌ Model not found at {model_path}")

        self.transform = transforms.Compose([
            transforms.Grayscale(1), transforms.Resize((64, 64)),
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
        ])

    def predict(self, roi_image):
        tensor = self.transform(roi_image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            conf, pred = torch.max(probs, 0)
        
        # Use dynamic class names from config
        label = cfg.class_names[pred.item()]
        return label, conf.item()