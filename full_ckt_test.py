import tkinter as tk
from tkinter import filedialog, messagebox, Menu
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import random
import threading
import glob

# --- CONFIGURATION ---
MODEL_PATH = "circuit_model_universal_all.pth"
DATASET_DIR = "DATA/dataset_closed_loop"  # Where we load random images from
CORRECTION_DIR = "DATA/dataset_corrected"  # Where we save cropped components (for classifier)
FULL_YOLO_DIR = "DATA/dataset_full_yolo"  # NEW: Where we save full images + labels (for detector)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COMPONENT_GROUPS = {
    "Basics": ["resistor", "capacitor", "inductor", "ground", "junction"],
    "Active": ["bjt", "mosfet", "diode", "led", "opamp"],
    "Sources": ["voltage_source", "current_source", "ac_source"],
    "Logic/Switch": ["switch", "logic_gate", "potentiometer", "transformer"],
    "Other": ["text", "wire"]
}

# Create a flattened list of classes for indexing
ALL_CLASSES = []
for cat in COMPONENT_GROUPS.values():
    for item in cat:
        if item not in ALL_CLASSES: ALL_CLASSES.append(item)
ALL_CLASSES.sort()

COLORS = {
    'default': '#00FF00',  # Green
    'selected':'#0000FF',  # Blue
    'wire':    '#00CCFF',  # Light Blue
    'text':    '#FF0000',  # Red
    'junction':'#FFFF00'   # Yellow
}

# --- 1. MODEL ARCHITECTURE ---
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
            nn.Linear(128 * 8 * 8, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

# --- 2. DATASET ---
class CorrectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        for idx, label in enumerate(ALL_CLASSES):
            folder = os.path.join(root_dir, label)
            if os.path.exists(folder):
                files = glob.glob(os.path.join(folder, "*.png"))
                for f in files:
                    self.images.append(f)
                    self.labels.append(idx)
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]
        if self.transform: image = self.transform(image)
        return image, label

# --- 3. BACKEND ---
class CircuitBackend:
    def __init__(self):
        self.model = CircuitNet(num_classes=len(ALL_CLASSES)).to(DEVICE)
        self.load_model_safe()
        self.transform = transforms.Compose([
            transforms.Grayscale(1), transforms.Resize((64, 64)),
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Setup directories for Crops (Classification)
        if not os.path.exists(CORRECTION_DIR): os.makedirs(CORRECTION_DIR)
        for c in ALL_CLASSES: os.makedirs(os.path.join(CORRECTION_DIR, c), exist_ok=True)
        
        # Setup directories for Full YOLO Data (Detection)
        os.makedirs(f"{FULL_YOLO_DIR}/images", exist_ok=True)
        os.makedirs(f"{FULL_YOLO_DIR}/labels", exist_ok=True)
        
        # Save a classes.txt so we know the order later
        with open(f"{FULL_YOLO_DIR}/classes.txt", "w") as f:
            for cls in ALL_CLASSES:
                f.write(f"{cls}\n")

    def load_model_safe(self):
        if not os.path.exists(MODEL_PATH): return
        saved_state = torch.load(MODEL_PATH, map_location=DEVICE)
        model_state = self.model.state_dict()
        pretrained_dict = {k: v for k, v in saved_state.items() if k in model_state and v.size() == model_state[k].size()}
        model_state.update(pretrained_dict)
        self.model.load_state_dict(model_state)
        self.model.eval()

    def detect(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return None, []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        mask = thresh - cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_k) - cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_k)
        mask = cv2.dilate(mask, np.ones((3,3),np.uint8), iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            pad = 10
            y1, y2 = max(0, y-pad), min(gray.shape[0], y+h+pad)
            x1, x2 = max(0, x-pad), min(gray.shape[1], x+w+pad)
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0 or w*h < 50: continue
            roi_t = self.transform(Image.fromarray(roi)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = self.model(roi_t)
                prob = torch.nn.functional.softmax(out[0], dim=0)
                conf, pred = torch.max(prob, 0)
            label = ALL_CLASSES[pred.item()]
            detections.append({'label': label, 'box': (x, y, w, h), 'conf': conf.item()})
        return img, detections

    def save_roi(self, image_cv2, box, label):
        """Saves individual crops (for classifier training)"""
        x, y, w, h = box
        y1, y2 = max(0, y-10), min(image_cv2.shape[0], y+h+10)
        x1, x2 = max(0, x-10), min(image_cv2.shape[1], x+w+10)
        roi = image_cv2[y1:y2, x1:x2]
        if len(roi.shape) == 3: roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        folder = os.path.join(CORRECTION_DIR, label)
        fname = f"train_{random.randint(100000,999999)}.png"
        cv2.imwrite(os.path.join(folder, fname), roi)

    def save_full_yolo(self, image_cv2, detections):
        """
        Saves the FULL image and a corresponding .txt file with 
        normalized coordinates for YOLO object detection training.
        """
        # Generate a unique ID for this image-label pair
        file_id = f"img_{random.randint(100000, 999999)}"
        
        # 1. Save the Image
        img_path = os.path.join(FULL_YOLO_DIR, "images", f"{file_id}.png")
        cv2.imwrite(img_path, image_cv2)
        
        # 2. Save the Labels (.txt)
        txt_path = os.path.join(FULL_YOLO_DIR, "labels", f"{file_id}.txt")
        
        h_img, w_img = image_cv2.shape[:2]
        
        with open(txt_path, "w") as f:
            for det in detections:
                label = det['label']
                
                # We only want to save known classes (skip '?' or unassigned)
                if label not in ALL_CLASSES:
                    continue
                
                class_id = ALL_CLASSES.index(label)
                x, y, w, h = det['box']
                
                # YOLO Format: class_id center_x center_y width height (All normalized 0-1)
                center_x = (x + w / 2) / w_img
                center_y = (y + h / 2) / h_img
                norm_w = w / w_img
                norm_h = h / h_img
                
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    def run_training(self, progress_callback):
        dataset = CorrectionDataset(CORRECTION_DIR, transform=self.transform)
        if len(dataset) == 0: return "No images in 'dataset_corrected'!"
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0005) 
        epochs = 10
        for epoch in range(epochs):
            total_loss = 0
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                optimizer.zero_grad()
                out = self.model(imgs)
                loss = criterion(out, lbls)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if progress_callback: progress_callback((epoch + 1) / epochs * 100)
            print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
        torch.save(self.model.state_dict(), MODEL_PATH)
        self.model.eval()
        return f"Training Complete on {len(dataset)} samples."

# --- 4. GUI ---
class CircuitEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Circuit AI Studio (Full Dataset Builder)")
        self.root.geometry("1400x900")
        self.backend = CircuitBackend()
        self.current_img = None 
        self.detections = []
        self.selected_indices = set()
        self.top_pad = 60
        self.scale = 1.0
        self.start_x = None
        self.start_y = None
        self.temp_rect = None

        self._init_layout()
        self.root.bind("<Delete>", self.delete_selection)

    def _init_layout(self):
        left = tk.Frame(self.root, bg="#222")
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, bg="#222", cursor="cross")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_click_start)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_click_release)
        self.canvas.bind("<Button-3>", self.on_right_click)

        right = tk.Frame(self.root, bg="white", width=350)
        right.pack(side="right", fill="y")
        
        tk.Label(right, text="AI Controls", font=("Arial", 16, "bold")).pack(pady=20)
        btn_frame = tk.Frame(right, bg="white")
        btn_frame.pack(fill="x", padx=20)
        tk.Button(btn_frame, text="🎲 Random", command=self.load_random, bg="#4CAF50", fg="white", width=15).grid(row=0, column=0, pady=5)
        tk.Button(btn_frame, text="📂 Upload", command=self.upload, bg="#2196F3", fg="white", width=15).grid(row=0, column=1, pady=5)
        
        leg_frame = tk.LabelFrame(right, text="Legend", bg="white")
        leg_frame.pack(fill="x", padx=20, pady=10)
        self._add_legend(leg_frame, "Selected", COLORS['selected'])
        self._add_legend(leg_frame, "Component", COLORS['default'])
        self._add_legend(leg_frame, "Text/Noise", COLORS['text'])

        data_frame = tk.LabelFrame(right, text="Dataset Actions", bg="#E3F2FD")
        data_frame.pack(fill="x", padx=20, pady=15)
        tk.Label(data_frame, text="Drag to Select Multiple.", bg="#E3F2FD").pack(anchor="w")
        tk.Button(data_frame, text="💾 Add All to Training", command=self.save_all_to_dataset, 
                  bg="#673AB7", fg="white", font=("Arial", 11)).pack(fill="x", padx=10, pady=5)

        train_frame = tk.LabelFrame(right, text="Model Training (Classifier)", bg="#fff8e1")
        train_frame.pack(fill="x", padx=20, pady=10)
        self.btn_train = tk.Button(train_frame, text="🔄 RETRAIN MODEL", command=self.start_training, 
                                   bg="#FF9800", fg="white", font=("Arial", 11, "bold"))
        self.btn_train.pack(fill="x", padx=10, pady=10)
        self.progress = Progressbar(train_frame, orient="horizontal", length=100, mode="determinate")
        self.progress.pack(fill="x", padx=10, pady=5)

    def _add_legend(self, parent, text, color):
        row = tk.Frame(parent, bg="white")
        row.pack(anchor="w", padx=5, pady=2)
        tk.Label(row, bg=color, width=3).pack(side="left")
        tk.Label(row, text=text, bg="white").pack(side="left", padx=5)

    def get_img_coords(self, cx, cy):
        if not hasattr(self, 'tk_img'): return 0, 0
        c_w, c_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        i_w, i_h = self.tk_img.width(), self.tk_img.height()
        off_x, off_y = (c_w - i_w) // 2, (c_h - i_h) // 2
        ix = (cx - off_x) / self.scale
        iy = (cy - off_y) / self.scale
        return ix, iy - self.top_pad

    def find_box_at(self, x_canvas, y_canvas):
        ix, iy = self.get_img_coords(x_canvas, y_canvas)
        for i in reversed(range(len(self.detections))):
            x, y, w, h = self.detections[i]['box']
            if x < ix < x+w and y < iy < y+h: return i
        return None

    def on_click_start(self, event):
        idx = self.find_box_at(event.x, event.y)
        
        if idx is not None:
            ctrl_pressed = (event.state & 0x0004)
            if not ctrl_pressed and idx not in self.selected_indices:
                self.selected_indices.clear()
            
            if ctrl_pressed:
                if idx in self.selected_indices: self.selected_indices.remove(idx)
                else: self.selected_indices.add(idx)
            else:
                self.selected_indices.add(idx)
            self.draw()
            return
        
        if not (event.state & 0x0004): self.selected_indices.clear()
        
        self.draw()
        self.start_x, self.start_y = event.x, event.y
        self.temp_rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="cyan", width=2, tag="drag")

    def on_drag(self, event):
        if self.temp_rect: self.canvas.coords(self.temp_rect, self.start_x, self.start_y, event.x, event.y)

    def on_click_release(self, event):
        if not self.temp_rect: return
        self.canvas.delete(self.temp_rect)
        self.temp_rect = None
        
        dist = abs(self.start_x - event.x) + abs(self.start_y - event.y)
        if dist < 5: return 

        x1, y1 = self.get_img_coords(self.start_x, self.start_y)
        x2, y2 = self.get_img_coords(event.x, event.y)
        rx1, rx2 = min(x1, x2), max(x1, x2)
        ry1, ry2 = min(y1, y2), max(y1, y2)

        found_any = False
        for i, det in enumerate(self.detections):
            bx, by, bw, bh = det['box']
            bx2, by2 = bx + bw, by + bh
            if (bx < rx2 and bx2 > rx1 and by < ry2 and by2 > ry1):
                self.selected_indices.add(i)
                found_any = True
        
        if found_any:
            self.draw()
            return

        box = (int(rx1), int(ry1), int(rx2-rx1), int(ry2-ry1))
        self.detections.append({'label': '?', 'box': box})
        self.selected_indices = {len(self.detections)-1}
        self.draw()
        self.popup_menu(event)

    def on_right_click(self, event):
        idx = self.find_box_at(event.x, event.y)
        if idx is not None:
            if idx not in self.selected_indices:
                self.selected_indices = {idx}
                self.draw()
        self.popup_menu(event)

    def popup_menu(self, event):
        menu = Menu(self.root, tearoff=0)
        def batch_set_label(l):
            for idx in self.selected_indices:
                self.detections[idx]['label'] = l
            self.draw()

        menu.add_command(label=f"Delete {len(self.selected_indices)} Item(s)", command=lambda: self.delete_selection(None))
        menu.add_separator()
        for cat, items in COMPONENT_GROUPS.items():
            sub = Menu(menu, tearoff=0)
            menu.add_cascade(label=cat, menu=sub)
            for item in items:
                sub.add_command(label=item.upper(), command=lambda i=item: batch_set_label(i))
        menu.post(event.x_root, event.y_root)

    def delete_selection(self, event):
        if not self.selected_indices: return
        for idx in sorted(list(self.selected_indices), reverse=True):
            del self.detections[idx]
        self.selected_indices.clear()
        self.draw()

    def save_all_to_dataset(self):
        if not self.detections or self.current_img is None: return
        
        # 1. Save Crops (Old method, for classifier)
        count = 0
        for det in self.detections:
            lbl = det['label']
            if lbl not in ['?', 'wire']:
                self.backend.save_roi(self.current_img, det['box'], lbl)
                count += 1
        
        # 2. Save FULL Image + Label File (New method, for YOLO)
        self.backend.save_full_yolo(self.current_img, self.detections)
        
        messagebox.showinfo("Saved", f"Saved {count} component crops.\nSaved FULL image to 'dataset_full_yolo'.")

    def process_image(self, path):
        img, dets = self.backend.detect(path)
        self.current_img = img
        self.detections = dets
        self.selected_indices.clear()
        self.draw()

    def draw(self):
        if self.current_img is None: return
        display = self.current_img.copy()
        display = cv2.copyMakeBorder(display, self.top_pad, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        for i, det in enumerate(self.detections):
            x, y, w, h = det['box']
            lbl = det['label']
            y_draw = y + self.top_pad
            
            if i in self.selected_indices:
                c_hex = COLORS['selected']
                thick = 4
            else:
                c_hex = COLORS.get(lbl, COLORS['default'])
                if lbl == 'text': c_hex = COLORS['text']
                elif lbl == 'wire': c_hex = COLORS['wire']
                elif lbl == 'junction': c_hex = COLORS['junction']
                thick = 2
            
            rgb = tuple(int(c_hex.lstrip('#')[k:k+2], 16) for k in (0, 2, 4))
            bgr = (rgb[2], rgb[1], rgb[0])
            
            cv2.rectangle(display, (x, y_draw), (x+w, y_draw+h), bgr, thick)
            
            if lbl not in ['wire', 'text', '?', 'junction']:
                txt = f"{lbl}"
                cv2.putText(display, txt, (x, y_draw-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)

        disp_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(disp_rgb)
        c_w = max(self.canvas.winfo_width(), 800)
        c_h = max(self.canvas.winfo_height(), 600)
        i_w, i_h = pil_img.size
        self.scale = min(c_w/i_w, c_h/i_h)
        pil_img = pil_img.resize((int(i_w*self.scale), int(i_h*self.scale)), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(c_w//2, c_h//2, image=self.tk_img, anchor="center")

    def load_random(self):
        if not os.path.exists(DATASET_DIR): return
        f = random.choice(os.listdir(DATASET_DIR))
        self.process_image(os.path.join(DATASET_DIR, f))

    def upload(self):
        p = filedialog.askopenfilename()
        if p: self.process_image(p)

    def start_training(self):
        self.btn_train.config(state="disabled", text="TRAINING...")
        self.progress['value'] = 0
        def train_thread():
            msg = self.backend.run_training(lambda p: self.root.after(0, lambda: self.progress.configure(value=p)))
            self.root.after(0, lambda: self.finish_training(msg))
        threading.Thread(target=train_thread).start()

    def finish_training(self, msg):
        messagebox.showinfo("Result", msg)
        self.btn_train.config(state="normal", text="🔄 RETRAIN MODEL")
        self.progress['value'] = 0

if __name__ == "__main__":
    root = tk.Tk()
    app = CircuitEditor(root)
    root.mainloop()