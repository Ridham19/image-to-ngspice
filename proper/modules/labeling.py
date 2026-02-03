import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import os
import json
import numpy as np

# Internal Imports
from modules.config import cfg
from modules.model import ComponentDetector
from modules.processing import preprocess_image, separate_layers
from modules.netlist import trace_nodes, generate_spice_text

class AnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Ridham19-AI: Labeling Tool")
        self.root.geometry("1400x900")
        
        self.current_img_path = None
        self.cv_image = None
        self.wire_mask = None
        self.components = []
        
        # View State
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.is_panning = False
        self.last_mouse = (0, 0)
        
        # Tools
        self.tool_mode = "select"
        self.selected_idx = None
        self.show_wires = tk.BooleanVar(value=True)
        self.start_x = None
        self.start_y = None
        
        self.out_dir = "dataset_new"
        self._init_dirs()
        self.detector = ComponentDetector("best.pt")
        
        self._setup_ui()

    def _init_dirs(self):
        for d in ["images", "labels", "meta"]:
            os.makedirs(os.path.join(self.out_dir, d), exist_ok=True)

    def _setup_ui(self):
        # Toolbar
        toolbar = tk.Frame(self.root, bg="#333", pady=5)
        toolbar.pack(fill="x")
        
        tk.Button(toolbar, text="📂 Open Image", command=self.load_image, bg="#4CAF50", fg="white").pack(side="left", padx=5)
        tk.Checkbutton(toolbar, text="Show Wires", variable=self.show_wires, command=self.redraw_canvas, bg="#333", fg="white", selectcolor="#555").pack(side="left", padx=10)
        
        tk.Label(toolbar, text="| Tools:", bg="#333", fg="#aaa").pack(side="left", padx=5)
        self.btn_sel = tk.Button(toolbar, text="👆 Select", command=lambda: self.set_tool("select"), bg="#666", fg="white")
        self.btn_sel.pack(side="left", padx=2)
        self.btn_box = tk.Button(toolbar, text="📦 New Box", command=lambda: self.set_tool("draw_box"), bg="#333", fg="white")
        self.btn_box.pack(side="left", padx=2)
        self.btn_wire = tk.Button(toolbar, text="✏️ Draw Wire", command=lambda: self.set_tool("draw_wire"), bg="#333", fg="white")
        self.btn_wire.pack(side="left", padx=2)
        self.btn_erase = tk.Button(toolbar, text="🧽 Erase Wire", command=lambda: self.set_tool("erase_wire"), bg="#333", fg="white")
        self.btn_erase.pack(side="left", padx=2)
        
        tk.Button(toolbar, text="💾 SAVE DATASET", command=self.save_data, bg="#2196F3", fg="white").pack(side="right", padx=20)

        # Main Layout
        main = tk.PanedWindow(self.root, orient="horizontal", bg="#222")
        main.pack(fill="both", expand=True)
        
        # Left Canvas
        self.canvas_frame = tk.Frame(main, bg="#111")
        self.canvas = tk.Canvas(self.canvas_frame, bg="#111", cursor="cross")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_left_down)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_up)
        self.canvas.bind("<Button-3>", self.start_pan)
        self.canvas.bind("<B3-Motion>", self.do_pan)
        self.canvas.bind("<MouseWheel>", self.do_zoom)
        main.add(self.canvas_frame, minsize=800)
        
        # Right Panel
        self.panel = tk.Frame(main, bg="#2d2d2d", width=300)
        main.add(self.panel, minsize=300)
        self._setup_editor(self.panel)

    def _setup_editor(self, parent):
        tk.Label(parent, text="Properties", bg="#2d2d2d", fg="white", font=("Arial", 12, "bold")).pack(pady=10)
        
        form = tk.Frame(parent, bg="#2d2d2d")
        form.pack(fill="x", padx=10)
        
        tk.Label(form, text="Class:", bg="#2d2d2d", fg="#aaa").grid(row=0, column=0, sticky="w")
        self.cb_class = ttk.Combobox(form, values=cfg.class_names, state="readonly")
        self.cb_class.grid(row=0, column=1, sticky="ew", pady=5)
        self.cb_class.bind("<<ComboboxSelected>>", self.update_selected)
        
        tk.Label(form, text="Value:", bg="#2d2d2d", fg="#aaa").grid(row=1, column=0, sticky="w")
        self.entry_val = tk.Entry(form, bg="#444", fg="white")
        self.entry_val.grid(row=1, column=1, sticky="ew", pady=5)
        self.entry_val.bind("<Return>", self.update_selected)
        
        tk.Label(form, text="Name:", bg="#2d2d2d", fg="#aaa").grid(row=2, column=0, sticky="w")
        self.lbl_name = tk.Label(form, text="-", bg="#2d2d2d", fg="white")
        self.lbl_name.grid(row=2, column=1, sticky="w")

        tk.Button(parent, text="Delete Component", command=self.delete_comp, bg="#C62828", fg="white").pack(fill="x", padx=10, pady=10)
        
        tk.Label(parent, text="Netlist Preview", bg="#2d2d2d", fg="white").pack(pady=(20,5))
        self.txt_netlist = tk.Text(parent, height=15, bg="#111", fg="#00FF00", font=("Consolas", 9))
        self.txt_netlist.pack(fill="both", expand=True, padx=10, pady=10)

    # --- CORE LOGIC ---
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if not path: return
        self.current_img_path = path
        
        self.cv_image, gray, binary = preprocess_image(path)
        _, self.wire_mask, _ = separate_layers(gray, binary)
        
        # Fit View
        h, w = self.cv_image.shape[:2]
        wh = self.canvas.winfo_height()
        self.zoom = min(1.0, (wh - 50) / h)
        self.offset_x, self.offset_y = 20, 20
        
        detections = self.detector.detect(self.cv_image)
        self.components = []
        counts = {k:0 for k in cfg.class_names}
        
        temp_comps = [{'box': d['box']} for d in detections]
        connections = trace_nodes(self.wire_mask, temp_comps)
        
        for i, det in enumerate(detections):
            label = det['label']
            if label in ['wire', 'text']: continue
            counts[label] += 1
            name = f"{cfg.get_prefix(label)}{counts[label]}"
            
            self.components.append({
                'label': label,
                'box': det['box'],
                'name': name,
                'nodes': connections[i],
                'value': '1k'
            })
            
        self.redraw_canvas()
        self.update_netlist_view()

    # --- DRAWING UTILS ---
    def screen_to_img(self, sx, sy):
        ix = int((sx - self.offset_x) / self.zoom)
        iy = int((sy - self.offset_y) / self.zoom)
        return ix, iy

    def redraw_canvas(self):
        if self.cv_image is None: return
        display = self.cv_image.copy()
        
        if self.show_wires.get() and self.wire_mask is not None:
            green = np.zeros_like(display)
            green[:, :, 1] = self.wire_mask
            mask = self.wire_mask > 0
            display[mask] = cv2.addWeighted(display[mask], 0.6, green[mask], 0.4, 0)

        for i, c in enumerate(self.components):
            x, y, w, h = c['box']
            col = (0, 255, 0) if i == self.selected_idx else (0, 100, 255)
            cv2.rectangle(display, (x, y), (x+w, y+h), col, 2)
            cv2.putText(display, c['name'], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5*self.zoom, col, 2)

        h, w = display.shape[:2]
        nh, nw = int(h*self.zoom), int(w*self.zoom)
        resized = cv2.resize(display, (nw, nh), interpolation=cv2.INTER_NEAREST)
        
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_image)

    # --- INTERACTION ---
    def start_pan(self, e): self.last_mouse = (e.x, e.y); self.is_panning = True
    def do_pan(self, e):
        if self.is_panning:
            self.offset_x += e.x - self.last_mouse[0]
            self.offset_y += e.y - self.last_mouse[1]
            self.last_mouse = (e.x, e.y)
            self.redraw_canvas()
    
    def do_zoom(self, e):
        factor = 0.9 if (e.num == 5 or e.delta < 0) else 1.1
        self.zoom *= factor
        self.redraw_canvas()

    def set_tool(self, t):
        self.tool_mode = t
        colors = {"select":"#333","draw_box":"#333","draw_wire":"#333","erase_wire":"#333"}
        colors[t] = "#666"
        self.btn_sel.config(bg=colors["select"])
        self.btn_box.config(bg=colors["draw_box"])
        self.btn_wire.config(bg=colors["draw_wire"])
        self.btn_erase.config(bg=colors["erase_wire"])

    def on_left_down(self, e):
        ix, iy = self.screen_to_img(e.x, e.y)
        self.start_x = ix; self.start_y = iy
        if self.tool_mode == "select": self.select_comp(ix, iy)
        elif "wire" in self.tool_mode: self.edit_wire(ix, iy)

    def on_left_drag(self, e):
        ix, iy = self.screen_to_img(e.x, e.y)
        if self.tool_mode == "draw_box":
            sx = (self.start_x * self.zoom) + self.offset_x
            sy = (self.start_y * self.zoom) + self.offset_y
            self.canvas.delete("temp")
            self.canvas.create_rectangle(sx, sy, e.x, e.y, outline="yellow", width=2, tags="temp")
        elif "wire" in self.tool_mode: self.edit_wire(ix, iy)

    def on_left_up(self, e):
        ix, iy = self.screen_to_img(e.x, e.y)
        if self.tool_mode == "draw_box":
            self.canvas.delete("temp")
            x1, y1 = min(self.start_x, ix), min(self.start_y, iy)
            w, h = abs(ix-self.start_x), abs(iy-self.start_y)
            if w>5 and h>5: self.add_comp(x1, y1, w, h)
        elif "wire" in self.tool_mode: self.retrace()

    def edit_wire(self, x, y):
        if self.wire_mask is None: return
        h, w = self.wire_mask.shape
        if 0<=x<w and 0<=y<h:
            col = 255 if self.tool_mode == "draw_wire" else 0
            rad = 3 if self.tool_mode == "draw_wire" else 10
            cv2.circle(self.wire_mask, (x, y), rad, col, -1)
            self.redraw_canvas()

    def select_comp(self, x, y):
        self.selected_idx = None
        for i, c in enumerate(self.components):
            bx, by, bw, bh = c['box']
            if bx<=x<=bx+bw and by<=y<=by+bh: self.selected_idx = i
        self.populate_editor(); self.redraw_canvas()

    def add_comp(self, x, y, w, h):
        self.components.append({'label':'resistor','box':(x,y,w,h),'name':'NEW','nodes':[],'value':'1k'})
        self.selected_idx = len(self.components)-1
        self.retrace(); self.populate_editor()

    def populate_editor(self):
        if self.selected_idx is None: self.lbl_name.config(text="-"); return
        c = self.components[self.selected_idx]
        self.cb_class.set(c['label'])
        self.entry_val.delete(0, tk.END); self.entry_val.insert(0, c['value'])
        self.lbl_name.config(text=c['name'])

    def update_selected(self, e=None):
        if self.selected_idx is None: return
        c = self.components[self.selected_idx]
        c['label'] = self.cb_class.get()
        c['value'] = self.entry_val.get()
        # Rename logic could go here
        self.redraw_canvas(); self.update_netlist_view()

    def delete_comp(self):
        if self.selected_idx is not None:
            del self.components[self.selected_idx]
            self.selected_idx = None
            self.redraw_canvas(); self.update_netlist_view()

    def retrace(self):
        if self.wire_mask is None: return
        t = [{'box': c['box']} for c in self.components]
        conns = trace_nodes(self.wire_mask, t)
        for i, c in enumerate(self.components): c['nodes'] = conns[i]
        self.update_netlist_view()

    def update_netlist_view(self):
        comps = [{'name': c['name'], 'type': c['label']} for c in self.components]
        conns = [c['nodes'] for c in self.components]
        txt = generate_spice_text(comps, conns)
        self.txt_netlist.delete('1.0', tk.END); self.txt_netlist.insert(tk.END, txt)

    def save_data(self):
        if not self.current_img_path: return
        name = os.path.splitext(os.path.basename(self.current_img_path))[0]
        cv2.imwrite(os.path.join(self.out_dir, "images", f"{name}.png"), self.cv_image)
        h, w = self.cv_image.shape[:2]
        with open(os.path.join(self.out_dir, "labels", f"{name}.txt"), "w") as f:
            for c in self.components:
                try:
                    cid = cfg.class_names.index(c['label'])
                    bx, by, bw, bh = c['box']
                    f.write(f"{cid} {(bx+bw/2)/w:.6f} {(by+bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}\n")
                except: pass
        with open(os.path.join(self.out_dir, "meta", f"{name}.json"), "w") as f:
            json.dump(self.components, f, indent=2)
        messagebox.showinfo("Saved", "Dataset saved!")

def run_labeler(parent=None):
    win = tk.Toplevel(parent) if parent else tk.Tk()
    app = AnnotationTool(win)
    if not parent: win.mainloop()