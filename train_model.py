import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import os
import json
import random

# --- CONFIGURATION ---
MODEL_SAVE_PATH = "circuit_model_universal_all.pth"
CLASS_LIST_PATH = "classes.json"
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define valid component names to look for
# (This prevents training on full circuit images or random system files)
TARGET_CLASSES = [
    "resistor", "capacitor", "inductor", "ground", "junction",
    "bjt", "mosfet", "diode", "led", "opamp",
    "voltage_source", "current_source", "ac_source",
    "switch", "logic_gate", "potentiometer", "transformer",
    "text", "wire"
]

# --- 1. Custom Recursive Dataset ---
class UniversalDataset(Dataset):
    def __init__(self, root_dir='.', transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(TARGET_CLASSES)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        print(f"🔍 Scanning '{os.path.abspath(root_dir)}' for component images...")
        
        # Walk through EVERY folder in the project
        for root, dirs, files in os.walk(root_dir):
            folder_name = os.path.basename(root).lower()
            
            # Check if this folder matches a known component class
            if folder_name in self.class_to_idx:
                class_idx = self.class_to_idx[folder_name]
                count = 0
                for file in files:
                    if file.endswith('.png'):
                        self.image_paths.append(os.path.join(root, file))
                        self.labels.append(class_idx)
                        count += 1
                if count > 0:
                    print(f"   -> Found {count} images in: {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('L') # Force Grayscale
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image to prevent crash, or handle better
            return torch.zeros((1, 64, 64)), label

# --- 2. Transforms (Matches GUI) ---
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.RandomRotation(15),       # Augmentation
    transforms.RandomHorizontalFlip(),   # Augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Handle lighting noise
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- 3. Model Architecture (Matched to GUI) ---
class CircuitNet(nn.Module):
    def __init__(self, num_classes): 
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512), 
            nn.ReLU(), 
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

# --- 4. Training Function ---
def train_model():
    # Load Data
    full_dataset = UniversalDataset(root_dir='.', transform=transform)
    
    if len(full_dataset) == 0:
        print("❌ No images found! Make sure you have folders named like 'resistor', 'capacitor', etc.")
        return

    print(f"✅ Total Training Images: {len(full_dataset)}")
    
    # Save Class List
    with open(CLASS_LIST_PATH, 'w') as f:
        json.dump(full_dataset.classes, f)

    # Split
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize
    model = CircuitNet(num_classes=len(full_dataset.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    print(f"\n🚀 Starting Training for {EPOCHS} epochs on {DEVICE}...")

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total if total > 0 else 0
        avg_loss = train_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  --> 💾 Saved Best Model")

    print(f"\n✅ DONE! Model saved to {MODEL_SAVE_PATH}")
    print(f"Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_model()