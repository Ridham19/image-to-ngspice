import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import os

# Internal imports from your modular structure
from modules.processing import preprocess_image, separate_layers, get_component_contours
from modules.model import ComponentDetector
from modules.netlist import trace_nodes, generate_spice_text
from modules.config import cfg

class SpiceGuiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ridham19-AI: Image to SPICE")
        self.root.geometry("1100x750")
        
        # --- State ---
        self.current_image_path = None
        # Ensure your model file is in the root directory
        self.detector = ComponentDetector("circuit_model.pth")
        
        self._setup_ui()

    def _setup_ui(self):
        # Top Control Bar
        top_bar = tk.Frame(self.root, pady=10, bg="#2d2d2d")
        top_bar.pack(side="top", fill="x")
        
        tk.Button(top_bar, text="📁 Load Circuit Image", command=self.load_image, 
                  padx=20, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), bd=0).pack(side="left", padx=20)
        
        self.btn_run = tk.Button(top_bar, text="⚡ Generate Netlist", command=self.process_circuit, 
                                 state="disabled", padx=20, font=("Arial", 10, "bold"), bd=0)
        self.btn_run.pack(side="left")

        # Main Layout (Image Preview | Netlist View)
        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Left Panel: Image Display
        self.img_panel = tk.Label(main_frame, text="No Image Loaded", bg="#252526", 
                                  fg="white", relief="sunken", bd=0)
        self.img_panel.pack(side="left", fill="both", expand=True)
        
        # Right Panel: SPICE Text Output
        right_panel = tk.Frame(main_frame, padx=10, bg="#1e1e1e")
        right_panel.pack(side="right", fill="both")
        
        tk.Label(right_panel, text="Generated SPICE Code:", font=("Arial", 10, "bold"), 
                 bg="#1e1e1e", fg="white").pack(anchor="w", pady=(0,5))
        self.txt_output = scrolledtext.ScrolledText(right_panel, width=45, height=35, 
                                                    bg="#252526", fg="#d4d4d4", 
                                                    insertbackground="white", font=("Consolas", 10))
        self.txt_output.pack(fill="both", expand=True)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if path:
            self.current_image_path = path
            pil_img = Image.open(path)
            pil_img.thumbnail((550, 550)) 
            tk_img = ImageTk.PhotoImage(pil_img)
            self.img_panel.config(image=tk_img, text="")
            self.img_panel.image = tk_img
            
            self.btn_run.config(state="normal", bg="#0078d7", fg="white")
            self.txt_output.delete('1.0', tk.END)

    def process_circuit(self):
        if not self.current_image_path: return
        
        try:
            # 1. Processing Pipeline
            original, gray, binary = preprocess_image(self.current_image_path)
            comp_mask, wire_mask, ai_input_gray = separate_layers(gray, binary)
            contours = get_component_contours(comp_mask)
            
            detected_comps = []
            counts = {k: 0 for k in cfg.class_names}
            
            # 2. Component Identification
            for (x, y, w, h) in contours:
                pad = 10
                roi = ai_input_gray[max(0, y-pad):min(gray.shape[0], y+h+pad), 
                                    max(0, x-pad):min(gray.shape[1], x+w+pad)]
                
                if roi.size == 0: continue
                label, conf = self.detector.predict(Image.fromarray(roi))
                
                if label in ['wire', 'text'] or conf < 0.60: continue
                
                counts[label] += 1
                prefix = cfg.specs[label]['prefix']
                name = f"{prefix}{counts[label]}"
                detected_comps.append({'name': name, 'type': label, 'box': (x, y, w, h)})
                
                # Visual Feedback
                cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(original, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 3. Trace & Generate
            connections = trace_nodes(wire_mask, detected_comps)
            spice_code = generate_spice_text(detected_comps, connections)
            
            # 4. Update UI
            self.txt_output.delete('1.0', tk.END)
            self.txt_output.insert(tk.END, spice_code)
            
            final_pil = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            final_pil.thumbnail((550, 550))
            tk_final = ImageTk.PhotoImage(final_pil)
            self.img_panel.config(image=tk_final)
            self.img_panel.image = tk_final
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def run_main_gui():
    root = tk.Tk()
    app = SpiceGuiApp(root)
    root.mainloop()