import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import os

# --- 1. Setup & Hyperparameters ---
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.001
DATA_PATH = "dataset_v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Data Loading with Augmentation ---
# We keep transforms simple since we already added noise in data_gen.py
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Centers pixel values around 0
])

full_dataset = torchvision.datasets.ImageFolder(root=DATA_PATH, transform=transform)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_set, test_set = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. Refactored CNN Architecture ---
class CircuitNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # Stabilizes training with noise
            nn.ReLU(),
            nn.MaxPool2d(2), # 64 -> 32
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32 -> 16
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 16 -> 8
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4), # Prevents overfitting to specific noise patterns
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# --- 4. Training Loop with Validation ---
model = CircuitNet(num_classes=len(full_dataset.classes)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_acc = 0.0

print(f"Starting training on {DEVICE}...")
for epoch in range(EPOCHS):
    # Training Phase
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

    # Validation Phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {acc:.2f}%")

    # Save the "Best" model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "circuit_model.pth")
        print(f"--> Best model saved with {acc:.2f}% accuracy")

print("\nTraining Complete!")