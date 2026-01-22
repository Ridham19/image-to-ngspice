import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import random
from modules.model import ComponentDetector
from modules.config import cfg  # - Import the dynamic config instead of CLASS_NAMES

class LabelingApp:
    def __init__(self, root, dataset_dir, model_path):
        self.root = root
        self.root.title("Manual Labeling / Verifier")
        self.dataset_dir = dataset_dir
        
        # AI Helper
        self.detector = ComponentDetector(model_path)
        
        # UI
        self.img_label = tk.Label(root)
        self.img_label.pack(pady=10)
        
        self.lbl_info = tk.Label(root, text="Press Next", font=("Arial", 12))
        self.lbl_info.pack()
        
        btn = tk.Button(root, text="Next Random Image", command=self.next_image)
        btn.pack(pady=20)
        
        self.next_image()

    def next_image(self):
        # Find random image folders using the dynamic config names
        # - Use cfg.class_names for filtering valid directories
        folders = [f for f in os.listdir(self.dataset_dir) 
                   if os.path.isdir(os.path.join(self.dataset_dir, f)) and f in cfg.class_names]
        
        if not folders: 
            messagebox.showwarning("Warning", "No valid dataset folders found!")
            return
        
        true_label = random.choice(folders)
        folder_path = os.path.join(self.dataset_dir, true_label)
        images = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
        
        if not images: return self.next_image()
        
        img_name = random.choice(images)
        path = os.path.join(folder_path, img_name)
        
        # Predict using the AI module
        pil_img = Image.open(path)
        pred_label, conf = self.detector.predict(pil_img.convert('L'))
        
        # Update UI Display
        display = pil_img.resize((200, 200))
        tk_img = ImageTk.PhotoImage(display)
        self.img_label.config(image=tk_img)
        self.img_label.image = tk_img
        
        status = "✅ MATCH" if true_label == pred_label else f"❌ MISMATCH (True: {true_label})"
        self.lbl_info.config(text=f"AI Says: {pred_label} ({conf:.0%})\n{status}")

def run_labeler(dataset_path="dataset_clean_v1"):
    root = tk.Tk()
    # Ensure model path is correct for your local setup
    app = LabelingApp(root, dataset_path, "circuit_model.pth") 
    root.mainloop()