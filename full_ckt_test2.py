import tkinter as tk
from tkinter import filedialog, messagebox, Menu, Text, Scrollbar
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import random
import glob
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = "best.pt"           
DATASET_DIR = "DATA/dataset_testing_clean"
FULL_YOLO_DIR = "DATA/dataset_full_yolo"

COLORS = {
    'default': '#00FF00', 'selected':'#0000FF', 'wire': '#00CCFF', 
    'text': '#FF0000', 'junction':'#FFFF00', 'ground': '#FF00FF'
}

COMPONENT_GROUPS = {
    "Basics": ["resistor", "capacitor", "inductor", "ground", "junction"],
    "Active": ["transistor", "diode", "source"],
    "Other": ["text", "wire"]
}

# --- BACKEND (THE BRAIN & LOGIC) ---
class CircuitBackend:
    def __init__(self):
        print(f"🧠 Loading YOLO model from {MODEL_PATH}...")
        try:
            self.model = YOLO(MODEL_PATH)
            self.class_names = self.model.names 
            print("✅ Model Loaded!")
        except Exception as e:
            print(f"❌ Error loading best.pt: {e}")
            self.model = None

        os.makedirs(f"{FULL_YOLO_DIR}/images", exist_ok=True)
        os.makedirs(f"{FULL_YOLO_DIR}/labels", exist_ok=True)

    def detect(self, image_path):
        if self.model is None: return None, []
        img = cv2.imread(image_path)
        if img is None: return None, []

        results = self.model.predict(img, conf=0.25)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = self.class_names[int(box.cls[0].item())]
                conf = float(box.conf[0].item())
                
                detections.append({
                    'label': label,
                    'box': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                    'conf': conf
                })
        return img, detections

    def generate_netlist(self, image_cv2, detections):
        """
        The Magic: Converts Image + Boxes -> SPICE Code
        """
        if image_cv2 is None: return "No Image Loaded."

        h, w = image_cv2.shape[:2]
        
        # 1. WIRE EXTRACTION
        # Convert to binary (black traces on white background)
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Create a "Wire Mask" by erasing components
        wire_mask = binary.copy()
        for det in detections:
            x, y, bw, bh = det['box']
            lbl = det['label']
            if lbl in ['wire', 'text']: continue # Don't erase wires!
            
            # Erase the box (Draw Black)
            pad = 5
            # Transistors need careful erasing to keep legs
            if lbl == 'transistor':
                 cv2.rectangle(wire_mask, (x+10, y+10), (x+bw-10, y+bh-10), 0, -1)
            else:
                 cv2.rectangle(wire_mask, (x+pad, y+pad), (x+bw-pad, y+bh-pad), 0, -1)

        # 2. FIND NODES (Connected Components)
        num_labels, labels_im = cv2.connectedComponents(wire_mask)
        
        # 3. CONNECT COMPONENTS TO NODES
        comp_nodes = []
        gnd_id = None
        
        # First pass: Identify Ground Node
        for det in detections:
            if det['label'] == 'ground':
                x, y, bw, bh = det['box']
                # Search area around ground symbol
                zone = labels_im[max(0, y-5):min(h, y+bh+5), max(0, x-5):min(w, x+bw+5)]
                nodes = np.unique(zone)
                for node in nodes:
                    if node != 0: gnd_id = node # Found it!
        
        spice_lines = ["* AI Generated Netlist", ".title Circuit Studio Export"]
        
        counts = {}
        
        for det in detections:
            lbl = det['label']
            if lbl in ['ground', 'text', 'wire', '?']: continue
            
            x, y, bw, bh = det['box']
            
            # Scan borders to find touching nodes
            borders = [
                labels_im[max(0, y-5):y+5, x:x+bw],       # Top
                labels_im[min(h, y+bh-5):min(h, y+bh+5), x:x+bw], # Bottom
                labels_im[y:y+bh, max(0, x-5):x+5],       # Left
                labels_im[y:y+bh, min(w, x+bw-5):min(w, x+bw+5)] # Right
            ]
            
            found_nodes = set()
            for b in borders:
                for n in np.unique(b):
                    if n != 0: found_nodes.add(n)
            
            nodes = list(found_nodes)
            
            # Formatting
            counts[lbl] = counts.get(lbl, 0) + 1
            name = f"{lbl[0].upper()}{counts[lbl]}"
            if lbl == 'transistor': name = f"Q{counts[lbl]}"
            
            # Values (Defaults)
            val = "1k"
            if lbl == 'capacitor': val = "10u"
            if lbl == 'source': val = "5V"
            
            # Map Ground (Node X -> 0)
            clean_nodes = []
            for n in nodes:
                clean_nodes.append('0' if n == gnd_id else str(n))
            
            # Fix floating pins
            while len(clean_nodes) < 2: clean_nodes.append('0')
            
            if lbl == 'transistor':
                # Emitter-Base-Collector logic is hard purely visually
                # We simply assign the 3 found nodes
                if len(clean_nodes) < 3: clean_nodes.append(clean_nodes[-1])
                line = f"{name} {clean_nodes[0]} {clean_nodes[1]} {clean_nodes[2]} 2N2222"
            else:
                line = f"{name} {clean_nodes[0]} {clean_nodes[1]} {val}"
                
            spice_lines.append(line)
            
        spice_lines.append(".tran 0.1m 20m")
        spice_lines.append(".end")
        
        return "\n".join(spice_lines)

    def save_data_for_training(self, image_cv2, detections):
        # ... (Same as before) ...
        # I'll preserve this functionality but hide it for brevity
        if not detections: return 0
        file_id = f"manual_{random.randint(100000, 999999)}"
        img_save_path = os.path.join(FULL_YOLO_DIR, "images", f"{file_id}.jpg")
        cv2.imwrite(img_save_path, image_cv2)
        txt_save_path = os.path.join(FULL_YOLO_DIR, "labels", f"{file_id}.txt")
        h_img, w_img = image_cv2.shape[:2]
        saved_count = 0
        with open(txt_save_path, "w") as f:
            for det in detections:
                label = det['label']
                class_id = -1
                for id, name in self.class_names.items():
                    if name == label:
                        class_id = id
                        break
                if class_id == -1: continue
                x, y, w, h = det['box']
                cx, cy = (x + w/2)/w_img, (y + h/2)/h_img
                nw, nh = w/w_img, h/h_img
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                saved_count += 1
        return saved_count

# --- GUI ---
class CircuitEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Circuit Studio Pro (AI -> SPICE)")
        self.root.geometry("1500x900") # Wider for code box
        
        self.backend = CircuitBackend()
        self.current_img = None 
        self.detections = []
        self.selected_indices = set()
        
        self.scale = 1.0
        self.start_x = None
        self.start_y = None
        self.temp_rect = None

        self._init_layout()
        self.root.bind("<Delete>", self.delete_selection)

    def _init_layout(self):
        # LEFT: Canvas
        left = tk.Frame(self.root, bg="#222")
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, bg="#222", cursor="cross")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_click_start)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_click_release)
        self.canvas.bind("<Button-3>", self.on_right_click)

        # RIGHT: Controls
        right = tk.Frame(self.root, bg="white", width=400) # Wider
        right.pack(side="right", fill="y")
        
        tk.Label(right, text="Control Panel", font=("Arial", 14, "bold"), bg="white").pack(pady=10)
        
        # 1. Image Loaders
        btn_frame = tk.Frame(right, bg="white")
        btn_frame.pack(fill="x", padx=10)
        tk.Button(btn_frame, text="🎲 Random Image", command=self.load_random, bg="#4CAF50", fg="white").pack(side="left", fill="x", expand=True, padx=2)
        tk.Button(btn_frame, text="📂 Upload", command=self.upload, bg="#2196F3", fg="white").pack(side="left", fill="x", expand=True, padx=2)

        # 2. SPICE GENERATOR (The Main Feature)
        spice_frame = tk.LabelFrame(right, text="Netlist Generation", bg="#FFF3E0", font=("Arial", 10, "bold"))
        spice_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        tk.Button(spice_frame, text="⚡ GENERATE SPICE", command=self.run_spice, 
                  bg="#FF9800", fg="white", font=("Arial", 12, "bold"), height=2).pack(fill="x", padx=5, pady=5)
        
        self.txt_code = Text(spice_frame, height=20, bg="#2d2d2d", fg="#00FF00", font=("Consolas", 10))
        self.txt_code.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 3. Legend (Small)
        leg_frame = tk.LabelFrame(right, text="Legend", bg="white")
        leg_frame.pack(fill="x", padx=10, pady=5)
        self._add_legend(leg_frame, "Component", COLORS['default'])
        self._add_legend(leg_frame, "Wire/Net", COLORS['wire'])
        
        # 4. Save Training Data
        tk.Button(right, text="💾 Save Corrections for Training", command=self.save_data, 
                  bg="#607D8B", fg="white").pack(fill="x", padx=10, pady=10)

    def _add_legend(self, parent, text, color):
        row = tk.Frame(parent, bg="white")
        row.pack(anchor="w", padx=5)
        tk.Label(row, bg=color, width=2, height=1).pack(side="left")
        tk.Label(row, text=text, bg="white", font=("Arial", 8)).pack(side="left", padx=5)

    def run_spice(self):
        if self.current_img is None: return
        self.txt_code.delete("1.0", tk.END)
        self.txt_code.insert(tk.END, "Analyzing connections...\n")
        self.root.update()
        
        # Generate!
        code = self.backend.generate_netlist(self.current_img, self.detections)
        
        self.txt_code.delete("1.0", tk.END)
        self.txt_code.insert(tk.END, code)

    # --- CANVAS & INTERACTION (Same as before) ---
    def get_img_coords(self, cx, cy):
        if not hasattr(self, 'tk_img'): return 0, 0
        c_w, c_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        i_w, i_h = self.tk_img.width(), self.tk_img.height()
        off_x, off_y = (c_w - i_w) // 2, (c_h - i_h) // 2
        ix = (cx - off_x) / self.scale
        iy = (cy - off_y) / self.scale
        return ix, iy

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
            if not ctrl_pressed and idx not in self.selected_indices: self.selected_indices.clear()
            if ctrl_pressed:
                if idx in self.selected_indices: self.selected_indices.remove(idx)
                else: self.selected_indices.add(idx)
            else: self.selected_indices.add(idx)
            self.draw()
            return
        if not (event.state & 0x0004): self.selected_indices.clear()
        self.draw()
        self.start_x, self.start_y = event.x, event.y
        self.temp_rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="cyan", width=2, dash=(2,2))

    def on_drag(self, event):
        if self.temp_rect: self.canvas.coords(self.temp_rect, self.start_x, self.start_y, event.x, event.y)

    def on_click_release(self, event):
        if not self.temp_rect: return
        self.canvas.delete(self.temp_rect)
        self.temp_rect = None
        if abs(self.start_x - event.x) < 5: return
        x1, y1 = self.get_img_coords(self.start_x, self.start_y)
        x2, y2 = self.get_img_coords(event.x, event.y)
        rx1, rx2 = min(x1, x2), max(x1, x2)
        ry1, ry2 = min(y1, y2), max(y1, y2)
        
        found = False
        for i, det in enumerate(self.detections):
            bx, by, bw, bh = det['box']
            if (bx < rx2 and bx+bw > rx1 and by < ry2 and by+bh > ry1):
                self.selected_indices.add(i)
                found = True
        if found: 
            self.draw()
            return
            
        box = (int(rx1), int(ry1), int(rx2-rx1), int(ry2-ry1))
        self.detections.append({'label': 'wire', 'box': box, 'conf': 1.0})
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
        def set_lbl(l):
            for idx in self.selected_indices: self.detections[idx]['label'] = l
            self.draw()
        menu.add_command(label="❌ Delete", command=lambda: self.delete_selection(None))
        menu.add_separator()
        for cat, items in COMPONENT_GROUPS.items():
            sub = Menu(menu, tearoff=0)
            menu.add_cascade(label=cat, menu=sub)
            for item in items: sub.add_command(label=item.upper(), command=lambda i=item: set_lbl(i))
        menu.post(event.x_root, event.y_root)

    def delete_selection(self, event):
        if not self.selected_indices: return
        for idx in sorted(list(self.selected_indices), reverse=True): del self.detections[idx]
        self.selected_indices.clear()
        self.draw()

    def load_random(self):
        if not os.path.exists(DATASET_DIR): return
        files = glob.glob(os.path.join(DATASET_DIR, "*.png")) + glob.glob(os.path.join(DATASET_DIR, "*.jpg"))
        if files: self.process_image(random.choice(files))

    def upload(self):
        p = filedialog.askopenfilename()
        if p: self.process_image(p)

    def process_image(self, path):
        img, dets = self.backend.detect(path)
        if img is None: return
        self.current_img = img
        self.detections = dets
        self.selected_indices.clear()
        self.draw()
        self.txt_code.delete("1.0", tk.END) # Clear old code

    def save_data(self):
        if self.current_img is None: return
        c = self.backend.save_data_for_training(self.current_img, self.detections)
        messagebox.showinfo("Saved", f"Saved {c} labels for training!")

    def draw(self):
        if self.current_img is None: return
        display = self.current_img.copy()
        for i, det in enumerate(self.detections):
            x, y, w, h = det['box']
            lbl = det['label']
            c = COLORS.get('selected', '#0000FF') if i in self.selected_indices else COLORS.get(lbl, '#00FF00')
            cv2.rectangle(display, (x, y), (x+w, y+h), tuple(int(c.lstrip('#')[k:k+2], 16) for k in (0, 2, 4))[::-1], 2)
            if lbl not in ['wire','text']: cv2.putText(display, lbl, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        pil_img = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        c_w, c_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.scale = min(c_w/pil_img.width, c_h/pil_img.height)
        pil_img = pil_img.resize((int(pil_img.width*self.scale), int(pil_img.height*self.scale)), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(c_w//2, c_h//2, image=self.tk_img, anchor="center")

if __name__ == "__main__":
    root = tk.Tk()
    app = CircuitEditor(root)
    root.mainloop()