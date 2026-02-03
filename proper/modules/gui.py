import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import os

# Internal Imports
from modules.processing import preprocess_image, separate_layers
from modules.model import ComponentDetector
from modules.netlist import trace_nodes, generate_spice_text
from modules.config import cfg
import modules.labeling as labeler

class SpiceGuiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ridham19-AI: Image to SPICE (YOLO)")
        self.root.geometry("1200x800")
        
        self.current_image_path = None
        self.detector = ComponentDetector("best.pt") 
        
        self._setup_ui()

    def _setup_ui(self):
        # Toolbar
        toolbar = tk.Frame(self.root, bg="#333", pady=10)
        toolbar.pack(side="top", fill="x")
        
        tk.Button(toolbar, text="📂 Load Image", command=self.load_image,
                  bg="#4CAF50", fg="white", font=("Segoe UI", 10, "bold"), padx=15).pack(side="left", padx=10)
        
        tk.Button(toolbar, text="🏷️ Open Labeler", command=self.open_labeler,
                  bg="#FF9800", fg="white", font=("Segoe UI", 10, "bold"), padx=15).pack(side="left", padx=10)
        
        self.btn_gen = tk.Button(toolbar, text="⚡ Generate Netlist", command=self.run_pipeline,
                                 bg="#0078D7", fg="white", font=("Segoe UI", 10, "bold"), padx=15, state="disabled")
        self.btn_gen.pack(side="left", padx=10)

        # Content
        content = tk.Frame(self.root, bg="#222")
        content.pack(fill="both", expand=True)
        
        self.panel_img = tk.Label(content, text="No Image", bg="black", fg="#888")
        self.panel_img.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        right_frame = tk.Frame(content, bg="#222", width=400)
        right_frame.pack(side="right", fill="y", padx=10, pady=10)
        right_frame.pack_propagate(False)
        
        tk.Label(right_frame, text="SPICE Netlist", bg="#222", fg="white", font=("Segoe UI", 12)).pack(anchor="w")
        self.txt_out = scrolledtext.ScrolledText(right_frame, bg="#333", fg="white", font=("Consolas", 11), insertbackground="white")
        self.txt_out.pack(fill="both", expand=True)

    def open_labeler(self):
        # Launch Labeler as a TOP LEVEL window (Pop up)
        labeler.run_labeler(self.root)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if path:
            self.current_image_path = path
            self.show_image(path)
            self.btn_gen.config(state="normal")
            self.txt_out.delete('1.0', tk.END)

    def show_image(self, path, cv_img=None):
        if cv_img is not None:
            pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.open(path)
            
        w, h = pil_img.size
        # Smart Resize to fit window
        aspect = w / h
        disp_h = 600
        disp_w = int(disp_h * aspect)
        pil_img = pil_img.resize((disp_w, disp_h))
        
        tk_img = ImageTk.PhotoImage(pil_img)
        self.panel_img.config(image=tk_img, text="")
        self.panel_img.image = tk_img

    def run_pipeline(self):
        if not self.current_image_path: return
        try:
            # 1. Preprocess (Get Wires)
            original, gray, binary = preprocess_image(self.current_image_path)
            _, wire_mask, _ = separate_layers(gray, binary)
            
            # 2. YOLO Detect (Get Components)
            detections = self.detector.detect(original)
            
            detected_comps = []
            counts = {k:0 for k in cfg.class_names}
            vis_img = original.copy()
            
            for det in detections:
                label = det['label']
                x, y, w, h = det['box']
                
                if label in ['wire', 'text']: continue
                
                # Naming Logic
                counts[label] = counts.get(label, 0) + 1
                prefix = cfg.get_prefix(label)
                name = f"{prefix}{counts[label]}"
                
                detected_comps.append({'name': name, 'type': label, 'box': (x, y, w, h)})
                
                # Draw on preview
                cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(vis_img, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 3. Connectivity
            connections = trace_nodes(wire_mask, detected_comps)
            spice_code = generate_spice_text(detected_comps, connections)
            
            self.txt_out.delete('1.0', tk.END)
            self.txt_out.insert(tk.END, spice_code)
            self.show_image(None, cv_img=vis_img)
            
            messagebox.showinfo("Done", f"Found {len(detected_comps)} components.")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            print(e)

def run_main_gui():
    root = tk.Tk()
    app = SpiceGuiApp(root)
    root.mainloop()