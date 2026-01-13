import tkinter as tk
from tkinter import messagebox
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import os
import json
import random

# --- Configuration ---
MODEL_PATH = "circuit_model_universal.pth"
CLASS_LIST_PATH = "classes.json"
DATASET_DIR = "dataset_clean_v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Model Architecture (Must Match Training) ---
class UniversalCircuitNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# --- 2. Helper Functions ---
def load_resources():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_LIST_PATH):
        return None, None

    with open(CLASS_LIST_PATH, 'r') as f:
        class_names = json.load(f)
    
    model = UniversalCircuitNet(n_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    return model, class_names

def get_random_image():
    if not os.path.exists(DATASET_DIR): return None, None
    
    folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
    if not folders: return None, None
    
    true_label = random.choice(folders)
    folder_path = os.path.join(DATASET_DIR, true_label)
    
    images = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    if not images: return get_random_image() # Retry
    
    img_name = random.choice(images)
    return os.path.join(folder_path, img_name), true_label

# --- 3. The GUI Application ---
class CircuitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Schemdraw Component Classifier")
        self.root.geometry("400x550")
        self.root.resizable(False, False)

        # Load Model
        self.model, self.class_names = load_resources()
        if self.model is None:
            messagebox.showerror("Error", "Model or Class List not found!\nRun generator and trainer first.")
            root.destroy()
            return

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # --- UI Elements ---
        
        # Title
        tk.Label(root, text="Component AI", font=("Helvetica", 16, "bold")).pack(pady=10)

        # Image Container
        self.img_frame = tk.Frame(root, bg="white", width=200, height=200, relief="sunken", borderwidth=2)
        self.img_frame.pack(pady=10)
        self.img_frame.pack_propagate(False) # Prevent shrinking
        
        self.img_label = tk.Label(self.img_frame, bg="white")
        self.img_label.pack(expand=True)

        # Stats Area
        self.stats_frame = tk.Frame(root)
        self.stats_frame.pack(pady=20)

        self.lbl_true = tk.Label(self.stats_frame, text="True Label: -", font=("Arial", 12))
        self.lbl_true.pack(anchor="w")

        self.lbl_pred = tk.Label(self.stats_frame, text="Predicted: -", font=("Arial", 12, "bold"))
        self.lbl_pred.pack(anchor="w")

        self.lbl_conf = tk.Label(self.stats_frame, text="Confidence: -", font=("Arial", 12))
        self.lbl_conf.pack(anchor="w")

        # Status Bar (Success/Fail)
        self.lbl_status = tk.Label(root, text="...", font=("Arial", 14, "bold"), fg="gray")
        self.lbl_status.pack(pady=5)

        # Button
        self.btn_next = tk.Button(root, text="Predict Random Image", command=self.predict_next, 
                                  bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), height=2)
        self.btn_next.pack(side="bottom", fill="x", padx=20, pady=20)

        # Run first prediction immediately
        self.predict_next()

    def predict_next(self):
        img_path, true_label = get_random_image()
        if not img_path:
            messagebox.showerror("Error", "Dataset is empty or missing!")
            return

        # 1. Process Image for GUI (Display)
        pil_img = Image.open(img_path)
        # Resize for display (make it bigger than 64x64 so we can see it)
        display_img = pil_img.resize((180, 180), Image.Resampling.NEAREST)
        tk_img = ImageTk.PhotoImage(display_img)
        
        # Update Image Label
        self.img_label.config(image=tk_img)
        self.img_label.image = tk_img # Keep reference

        # 2. Process Image for AI (Inference)
        input_img = pil_img.convert('L')
        tensor = self.transform(input_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, idx = torch.max(probs, 0)

        pred_label = self.class_names[idx.item()]
        score = conf.item() * 100

        # 3. Update Text Stats
        self.lbl_true.config(text=f"True Label: {true_label}")
        self.lbl_pred.config(text=f"Predicted:  {pred_label}")
        self.lbl_conf.config(text=f"Confidence: {score:.2f}%")

        # 4. Color Coding
        if true_label == pred_label:
            self.lbl_status.config(text="✓ CORRECT", fg="green")
            self.lbl_pred.config(fg="green")
        else:
            self.lbl_status.config(text="✗ WRONG", fg="red")
            self.lbl_pred.config(fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = CircuitApp(root)
    root.mainloop()