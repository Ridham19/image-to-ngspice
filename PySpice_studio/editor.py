import tkinter as tk
from tkinter import filedialog, messagebox, Menu
import copy
import subprocess
import os
import sys
import json

# --- AI PIPELINE INTEGRATION ---
# Add the 'proper' directory to the system path so we can import the YOLO model
ai_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'proper'))
if ai_path not in sys.path:
    sys.path.append(ai_path)

try:
    from modules.model import ComponentDetector
    from modules.processing import preprocess_image, separate_layers
    from modules.netlist import trace_nodes
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("⚠️ Warning: Could not load AI modules. Make sure the 'proper' folder is situated correctly.")

# Internal module imports
from components import Component, ComponentHelper
from netlist import generate_netlist, analyze_circuit
from simulation_dialog import SimulationDialog
from library import DB 

# --- UI CONSTANTS & DARK THEME ---
CONFIG_FILE = "config.json"
GRID_SIZE = 20 # Don't change

COLOR_CANVAS_BG   = "#1E1E1E" 
COLOR_TOOLBAR_BG  = "#2D2D2D" 
COLOR_SIDEBAR_BG  = "#252526" 
COLOR_TEXT_LIGHT  = "#E0E0E0" 
COLOR_ACCENT_BLUE = "#0078D7" 
COLOR_WIRE        = "#4FC1FF" 
COLOR_GRID_DOT    = "#444444" 

class CircuitEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("PySpice Studio - Professional")
        
        try: self.root.state('zoomed')
        except: self.root.geometry("1400x900")

        self.ngspice_path = "ngspice"
        self.load_config()
        
        self.components = []
        self.wires = []
        self.sim_data = {'cmd': '.op', 'plots': {}, 'colors': {}} 
        
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.mode = "select"  
        self.counts = {}      
        
        self.selected_comps = []
        self.selected_wires = []
        self.clipboard = []
        
        self.selection_box_start = None
        self.drag_start_world = None
        self.wire_start = None
        self.ghost_rotation = 0
        self.hovered_pin = None

        self._setup_main_layout()
        self._setup_shortcuts()
        self._draw_grid()
        self.redraw_all()

    # ==========================================
    # LAYOUT & UI SETUP
    # ==========================================

    def _setup_main_layout(self):
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)

        left_frame = tk.Frame(self.root, bg=COLOR_TOOLBAR_BG)
        left_frame.grid(row=0, column=0, sticky="nsew")
        left_frame.rowconfigure(1, weight=1)
        left_frame.columnconfigure(0, weight=1)

        self.prop_frame = tk.Frame(self.root, width=300, bg=COLOR_SIDEBAR_BG, bd=0)
        self.prop_frame.grid(row=0, column=1, sticky="ns")
        self.prop_frame.pack_propagate(False)
        
        self._init_menu_bar()
        self._add_sidebar_header("PROPERTIES")
        self.prop_container = tk.Frame(self.prop_frame, bg=COLOR_SIDEBAR_BG, padx=10, pady=10)
        self.prop_container.pack(fill="x")

        self._add_sidebar_header("SIMULATION")
        self.lbl_sim = tk.Label(self.prop_frame, text=".op", bg=COLOR_SIDEBAR_BG, fg="#4CAF50", wraplength=280)
        self.lbl_sim.pack(pady=5)
        
        self._add_sidebar_footer_shortcuts()
        self._init_professional_toolbar(left_frame)
        self._init_canvas(left_frame)

    def _init_menu_bar(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        
        file_menu.add_command(label="📸 Import from Image (AI)...", command=self.import_from_image)
        file_menu.add_command(label="📄 Import AI JSON...", command=self.import_from_ai)
        file_menu.add_separator()
        file_menu.add_command(label="Settings (Set ngspice path)", command=self.open_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

    def _add_sidebar_header(self, text):
        tk.Label(self.prop_frame, text=text, bg="#333", fg=COLOR_TEXT_LIGHT, font=("Segoe UI", 9, "bold")).pack(fill="x", pady=(10, 0))

    def _add_sidebar_footer_shortcuts(self):
        info = "SHORTCUTS:\n[W] Wire   [P] Probe\n[R] Resistor [C] Cap\n[L] Inductor [D] Diode\n[Del] Delete\n[Ctrl+C/V] Copy/Paste\n[Ctrl+R] Rotate"
        tk.Label(self.prop_frame, text=info, bg=COLOR_SIDEBAR_BG, fg="#888", justify="left", font=("Consolas", 8)).pack(side="bottom", pady=20)

    def _setup_shortcuts(self):
        keys = {'w': 'wire', 'p': 'probe', 'r': 'resistor', 'c': 'capacitor', 'l': 'inductor', 'd': 'diode'}
        for key, mode in keys.items(): self.root.bind(key, lambda e, m=mode: self.set_mode(m))
        self.root.bind('<Delete>', self.delete_selection)
        self.root.bind('<Control-r>', self.rotate_command)
        self.root.bind('<Control-c>', self.copy_selection)
        self.root.bind('<Control-v>', self.paste_selection)

    # ==========================================
    # FILE, CONFIG & AI IMPORT LOGIC
    # ==========================================

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    self.ngspice_path = json.load(f).get("ngspice_path", "ngspice")
            except: pass

    def open_settings(self):
        path = filedialog.askopenfilename(title="Locate ngspice.exe", filetypes=[("Executable", "*.exe"), ("All", "*.*")])
        if path:
            self.ngspice_path = path
            with open(CONFIG_FILE, 'w') as f: json.dump({"ngspice_path": path}, f)
            messagebox.showinfo("Config Updated", f"Simulator path set to:\n{path}")

    def import_from_image(self):
        if not AI_AVAILABLE:
            messagebox.showerror("AI Engine Missing", "Could not load the YOLO modules. Check your folder structure.")
            return

        img_path = filedialog.askopenfilename(title="Select Hand-Drawn Circuit", filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")])
        if not img_path: return

        try:
            self.status.config(text="🤖 AI is processing image... Please wait.")
            self.root.update()

            # 1. Run YOLO Object Detection (Skip OpenCV completely!)
            detector = ComponentDetector()
            json_output_path = os.path.join(os.path.dirname(__file__), "detected_components.json")
            detected_comps = detector.detect(img_path, output_file=json_output_path)

            # 2. Native Tkinter AI Debug Window
            import cv2
            from PIL import Image, ImageTk
            
            vis_img = cv2.imread(img_path)
            for comp in detected_comps:
                x, y, w, h = comp['box']
                name = comp['name']
                cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                val = comp.get('value')
                if val and val != "TEXT_FOUND" and 'text_box' in comp:
                    tx, ty, tw, th = comp['text_box']
                    cv2.rectangle(vis_img, (tx, ty), (tx+tw, ty+th), (0, 165, 255), 2)
                    cv2.putText(vis_img, f"OCR: {val}", (tx, ty-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.putText(vis_img, name, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            rgb_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)

            im_w, im_h = pil_img.size
            if im_h > 700 or im_w > 1000:
                scale = min(1000/im_w, 700/im_h)
                pil_img = pil_img.resize((int(im_w*scale), int(im_h*scale)), Image.Resampling.LANCZOS)
                
            self.status.config(text="Review the AI detection window...")
            self.root.update()
            
            debug_win = tk.Toplevel(self.root)
            debug_win.title("AI Vision Debug")
            debug_win.configure(bg="#1E1E1E")
            
            tk_img = ImageTk.PhotoImage(pil_img)
            lbl = tk.Label(debug_win, image=tk_img, bg="#1E1E1E")
            lbl.image = tk_img 
            lbl.pack(padx=20, pady=20)
            
            tk.Button(debug_win, text="Accept & Load Circuit", command=debug_win.destroy, 
                      bg="#0078D7", fg="white", font=("Segoe UI", 12, "bold"), padx=20, pady=10).pack(pady=(0, 20))
            
            self.root.wait_window(debug_win)
            
            # 3. Load Data & Trigger Grid-Raycast Auto-Router
            self._load_ai_data_to_canvas(detected_comps)
            
            self.status.config(text="Ready")
            messagebox.showinfo("AI Import Success", f"AI placed components and auto-routed wires!")

        except Exception as e:
            self.status.config(text="Ready")
            messagebox.showerror("AI Processing Error", f"Failed to process image:\n{e}")

    def import_from_ai(self):
        filepath = filedialog.askopenfilename(title="Select AI JSON", filetypes=[("JSON Files", "*.json")])
        if not filepath: return
        try:
            with open(filepath, 'r') as f: ai_data = json.load(f)
            self._load_ai_data_to_canvas(ai_data)
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to load AI JSON:\n{e}")

    def _load_ai_data_to_canvas(self, ai_data):
        """Translates AI JSON array into Canvas Components AND Mathematically Routes Wires."""
        self.components = []
        self.wires = []
        self.selected_comps = []
        self.selected_wires = []
        
        type_mapping = {'voltage': 'source', 'ground': 'gnd', 'transistor': 'bjt_npn'}

        # 1. Place all components snapped perfectly to the grid
        for item in ai_data:
            raw_type = item['type']
            if raw_type in ['wire', 'junction', 'text']: continue 

            sp_type = type_mapping.get(raw_type, raw_type)
            cx, cy = item['center']
            
            snapped_x = round(cx / GRID_SIZE) * GRID_SIZE
            snapped_y = round(cy / GRID_SIZE) * GRID_SIZE

            rotation = 0
            if 'box' in item:
                x, y, w, h = item['box']
                if h > w * 1.2: rotation = 90

            comp = Component(sp_type, snapped_x, snapped_y, item['name'])
            comp.rotation = rotation

            detected_val = item.get('value')
            if detected_val:
                comp.value = "NEEDS_OCR" if detected_val == "TEXT_FOUND" else detected_val
                if comp.params and detected_val != "TEXT_FOUND":
                    first_key = list(comp.params.keys())[0]
                    comp.params[first_key] = detected_val

            self.components.append(comp)

        # 2. THE GRID-RAYCAST AUTO-ROUTER
        # Since components are on a grid, we check if their pins align perfectly horizontally or vertically
        all_pins = []
        for comp in self.components:
            for px, py in comp.get_pins():
                all_pins.append((px, py, comp))
                
        for i in range(len(all_pins)):
            for j in range(i + 1, len(all_pins)):
                x1, y1, c1 = all_pins[i]
                x2, y2, c2 = all_pins[j]
                
                if c1 == c2: continue # Don't short circuit a component to itself
                
                is_vertical_align = abs(x1 - x2) < 5
                is_horizontal_align = abs(y1 - y2) < 5
                
                if is_vertical_align or is_horizontal_align:
                    # Check if the distance is reasonable (don't connect across the whole page)
                    dist = abs(y1 - y2) if is_vertical_align else abs(x1 - x2)
                    if dist > 500: continue
                    
                    # Ensure no other component is blocking the path of this wire
                    blocked = False
                    for block_comp in self.components:
                        if block_comp in [c1, c2]: continue
                        
                        # If a component's center is sitting right on the wire's path, it's blocked
                        if is_vertical_align:
                            if abs(block_comp.x - x1) < 25 and min(y1, y2) < block_comp.y < max(y1, y2):
                                blocked = True; break
                        else:
                            if abs(block_comp.y - y1) < 25 and min(x1, x2) < block_comp.x < max(x1, x2):
                                blocked = True; break
                                
                    if not blocked:
                        # Success! Draw the wire perfectly straight between the pins
                        self.wires.append(((x1, y1), (x2, y2)))

        self.redraw_all()

    # ==========================================
    # SIMULATION ENGINE
    # ==========================================

    def open_sim_dialog(self): 
        node_map, sources, unique_nodes, sweepables = analyze_circuit(self.components, self.wires)
        SimulationDialog(self.root, unique_nodes, sources, sweepables, self.sim_data, self.set_sim_data)

    def set_sim_data(self, data): 
        self.sim_data = data
        self.lbl_sim.config(text=data['cmd'])

    def run_simulation(self):
        cwd = os.getcwd()
        filepath = os.path.join(cwd, "circuit.cir")
        netlist_code = generate_netlist(self.components, self.wires, self.sim_data)
        
        with open(filepath, "w") as f: f.write(netlist_code)
        
        if not self.ngspice_path or self.ngspice_path == "ngspice": 
            messagebox.showwarning("Warning", "ngspice path not configured. Using system default.")
        
        try: subprocess.Popen([self.ngspice_path, filepath], cwd=cwd)
        except Exception as e: messagebox.showerror("Execution Error", f"Failed to run simulation:\n{e}")

    # ==========================================
    # SIDEBAR PROPERTY EDITOR
    # ==========================================

    def update_sidebar(self):
        for widget in self.prop_container.winfo_children(): widget.destroy()
        
        if len(self.selected_comps) == 1: self._build_component_editor(self.selected_comps[0])
        elif self.selected_wires: tk.Label(self.prop_container, text=f"{len(self.selected_wires)} Wires Selected", bg=COLOR_SIDEBAR_BG, fg="white").pack(pady=20)
        else: tk.Label(self.prop_container, text="Select an object to edit", bg=COLOR_SIDEBAR_BG, fg="#888").pack(pady=20)

    def _build_component_editor(self, comp):
        self._add_prop_label("Reference ID:") 
        self.entry_name = self._add_prop_entry(comp.name)
        
        self.param_entries = {}
        for key, value in comp.params.items():
            self._add_prop_label(f"{key.upper()}:")
            self.param_entries[key] = self._add_prop_entry(str(value))
        
        tk.Button(self.prop_container, text="Update Component", command=self.apply_properties, bg=COLOR_ACCENT_BLUE, fg="white", relief="flat").pack(fill="x", pady=10)

    def _add_prop_label(self, text): tk.Label(self.prop_container, text=text, bg=COLOR_SIDEBAR_BG, fg="white", anchor="w").pack(fill="x")
    
    def _add_prop_entry(self, initial_val):
        e = tk.Entry(self.prop_container, bg="#444", fg="white", insertbackground="white", bd=0, highlightthickness=1)
        e.insert(0, initial_val)
        e.pack(fill="x", pady=(0, 10), ipady=2)
        e.bind("<Return>", self.apply_properties)
        return e

    def apply_properties(self, event=None):
        if len(self.selected_comps) == 1:
            comp = self.selected_comps[0]
            comp.name = self.entry_name.get()
            for key, entry in self.param_entries.items(): comp.params[key] = entry.get()
            self.redraw_all()

    # ==========================================
    # TOOLBAR & CANVAS VISUALS
    # ==========================================

    def _init_professional_toolbar(self, parent):
        ribbon = tk.Frame(parent, bg=COLOR_TOOLBAR_BG) 
        ribbon.grid(row=0, column=0, sticky="ew")

        def create_btn_group(title):
            f = tk.LabelFrame(ribbon, text=title, bg=COLOR_TOOLBAR_BG, fg="#AAA", font=("Segoe UI", 8), padx=5, pady=2)
            f.pack(side="left", fill="y", padx=5, pady=5)
            return f

        groups = {}
        for c_type, data in DB.items():
            cat = data['category']
            if cat not in groups: groups[cat] = create_btn_group(cat)
            btn_lbl = data.get('btn_text', data['label']) 
            tk.Button(groups[cat], text=btn_lbl, command=lambda t=c_type: self.set_mode(t), relief="flat", bg="#444", fg="white", font=("Segoe UI", 9)).pack(side="left", padx=2, pady=2)

        g_tools = create_btn_group("Drawing Tools")
        tk.Button(g_tools, text="✏️ WIRE", command=lambda: self.set_mode('wire'), bg=COLOR_ACCENT_BLUE, fg="white", relief="flat").pack(side="left", padx=2)
        tk.Button(g_tools, text="🔍 PROBE", command=lambda: self.set_mode("probe"), bg="#FF9800", fg="white", relief="flat").pack(side="left", padx=2)
        
        g_sim = create_btn_group("Simulation Control")
        tk.Button(g_sim, text="Config", command=self.open_sim_dialog, bg="#555", fg="white").pack(side="left", padx=2)
        tk.Button(g_sim, text="RUN", command=self.run_simulation, bg="#2E7D32", fg="white", font=("Segoe UI", 9, "bold")).pack(side="left", padx=5)
        
        self.status = tk.Label(ribbon, text="Ready", bg=COLOR_TOOLBAR_BG, fg="#888")
        self.status.pack(side="right", padx=20)

    def _init_canvas(self, parent):
        self.canvas = tk.Canvas(parent, bg=COLOR_CANVAS_BG, highlightthickness=0, cursor="cross")
        self.canvas.grid(row=1, column=0, sticky="nsew")
        
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Button-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.do_pan)
        self.canvas.bind("<MouseWheel>", self.do_zoom)

    # ==========================================
    # CORE INTERACTION LOGIC
    # ==========================================

    def on_left_click(self, event):
        self.canvas.focus_set()
        wx, wy = self.screen_to_world(event.x, event.y)
        
        if self.mode == 'probe': return self._handle_probe_click(wx, wy)
        if self.mode == 'wire': return self._handle_wire_click(wx, wy)
        if self.mode != 'select': return self.create_component(self.mode, wx, wy, self.ghost_rotation)
        
        self._handle_selection_click(wx, wy)

    def _handle_probe_click(self, wx, wy):
        clicked_comp = next((c for c in self.components if abs(c.x-wx)<30 and abs(c.y-wy)<30), None)
        node_map, _, _, _ = analyze_circuit(self.components, self.wires)
        snap_pt = (round(wx/GRID_SIZE)*GRID_SIZE, round(wy/GRID_SIZE)*GRID_SIZE)
        
        signal = None
        if clicked_comp and clicked_comp.type in ['source', 'current', 'ac_source', 'pulse', 'sine_source']:
            signal = f"i({clicked_comp.name})"
        elif snap_pt in node_map:
            signal = f"v({node_map[snap_pt]})"
            
        if signal:
            if 'plots' not in self.sim_data: self.sim_data['plots'] = {'1': []}
            if '1' not in self.sim_data['plots']: self.sim_data['plots']['1'] = []
            if signal not in self.sim_data['plots']['1']:
                self.sim_data['plots']['1'].append(signal)
                messagebox.showinfo("Probe", f"Added {signal} to Plot Window 1")
                total = sum(len(v) for v in self.sim_data['plots'].values())
                self.lbl_sim.config(text=f"{self.sim_data['cmd']} (Plot: {total})")

    def _handle_wire_click(self, wx, wy):
        if self.hovered_pin: wx, wy = self.hovered_pin
        if self.wire_start:
            sx, sy = self.wire_start
            if sx != wx: self.wires.append(((sx, sy), (wx, sy)))
            if sy != wy: self.wires.append(((wx, sy), (wx, wy)))
            self.redraw_all()
            self.wire_start = None if self.hovered_pin else (wx, wy)
        else:
            self.wire_start = (wx, wy)

    def _handle_selection_click(self, wx, wy):
        clicked_comp = next((c for c in self.components if abs(c.x-wx)<30 and abs(c.y-wy)<30), None)
        if clicked_comp:
            self.selected_comps, self.selected_wires = [clicked_comp], []
            self.drag_start_world = (wx, wy)
        else:
            clicked_wire = None
            for w in self.wires:
                (x1, y1), (x2, y2) = w
                if min(x1,x2)-5 <= wx <= max(x1,x2)+5 and min(y1,y2)-5 <= wy <= max(y1,y2)+5:
                    if (x1==x2 and abs(wx-x1)<5) or (y1==y2 and abs(wy-y1)<5):
                        clicked_wire = w; break
            if clicked_wire:
                self.selected_wires, self.selected_comps = [clicked_wire], []
            else:
                self.selected_comps, self.selected_wires = [], []
                self.selection_box_start = (wx, wy)
        
        self.update_sidebar()
        self.redraw_all()

    def on_drag(self, event):
        wx, wy = self.screen_to_world(event.x, event.y)
        if self.mode == 'select' and self.selection_box_start:
            self._draw_selection_rectangle(wx, wy)
        elif self.mode == 'select' and self.selected_comps and self.drag_start_world:
            dx, dy = wx - self.drag_start_world[0], wy - self.drag_start_world[1]
            if dx != 0 or dy != 0:
                for c in self.selected_comps: c.x += dx; c.y += dy
                self.drag_start_world = (wx, wy)
                self.redraw_all()

    def _draw_selection_rectangle(self, wx, wy):
        self.canvas.delete("sel_box")
        s_start = self.world_to_screen(*self.selection_box_start)
        s_end = self.world_to_screen(wx, wy)
        self.canvas.create_rectangle(s_start[0], s_start[1], s_end[0], s_end[1], outline=COLOR_ACCENT_BLUE, dash=(2,2), tags="sel_box")

    def on_release(self, event):
        wx, wy = self.screen_to_world(event.x, event.y)
        if self.mode == 'select' and self.selection_box_start:
            x1, y1 = self.selection_box_start; x2, y2 = wx, wy
            bx, by, bX, bY = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
            
            self.selected_comps = [c for c in self.components if bx <= c.x <= bX and by <= c.y <= bY]
            self.selected_wires = [w for w in self.wires if bx <= w[0][0] <= bX and by <= w[0][1] <= bY and bx <= w[1][0] <= bX and by <= w[1][1] <= bY]

            self.selection_box_start = None
            self.canvas.delete("sel_box")
            self.update_sidebar()
            self.redraw_all()
        self.drag_start_world = None

    # ==========================================
    # DRAWING UTILITIES
    # ==========================================

    def redraw_all(self):
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_wires()
        for comp in self.components: self._draw_component_visual(comp)

    def _draw_grid(self):
        if self.zoom < 0.4: return
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        swx, swy = self.screen_to_world(0,0); ewx, ewy = self.screen_to_world(w,h)
        step = GRID_SIZE if self.zoom > 0.7 else GRID_SIZE*2
        for x in range(int(swx), int(ewx), step):
            for y in range(int(swy), int(ewy), step):
                sx, sy = self.world_to_screen(x,y)
                self.canvas.create_rectangle(sx, sy, sx+1, sy+1, fill=COLOR_GRID_DOT, outline="")

    def _draw_wires(self):
        width = max(1, int(2*self.zoom))
        for wire in self.wires:
            col = COLOR_ACCENT_BLUE if wire in self.selected_wires else COLOR_WIRE
            p1, p2 = self.world_to_screen(*wire[0]), self.world_to_screen(*wire[1])
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=col, width=width)
            
            r = 3*self.zoom
            self.canvas.create_oval(p1[0]-r, p1[1]-r, p1[0]+r, p1[1]+r, fill=col, outline="")
            self.canvas.create_oval(p2[0]-r, p2[1]-r, p2[0]+r, p2[1]+r, fill=col, outline="")

    def _draw_component_visual(self, comp):
        sx, sy = self.world_to_screen(comp.x, comp.y)
        tk_img, _ = ComponentHelper.render_image(comp.type, comp.rotation, self.zoom)
        if not tk_img: return
        
        comp.img_ref = tk_img
        self.canvas.create_image(sx, sy, image=tk_img, anchor="center")
        
        if comp in self.selected_comps:
            w, h = tk_img.width(), tk_img.height()
            self.canvas.create_rectangle(sx-w/2-5, sy-h/2-5, sx+w/2+5, sy+h/2+5, outline=COLOR_ACCENT_BLUE, width=2, dash=(4,2))
        
        off = 35*self.zoom
        tx, ty = (sx+off, sy) if comp.rotation in [90, 270] else (sx, sy+off)
        self.canvas.create_text(tx, ty, text=f"{comp.name}\n{comp.value}", fill=COLOR_TEXT_LIGHT, font=("Arial", max(6, int(8*self.zoom))))
        
        r = 3*self.zoom
        for px, py in comp.get_pins():
            psx, psy = self.world_to_screen(px, py)
            self.canvas.create_rectangle(psx-r, psy-r, psx+r, psy+r, fill="red", outline="black")

    # ==========================================
    # STATE MODIFIERS
    # ==========================================

    def create_component(self, c_type, x, y, rotation):
        prefix = DB[c_type]['prefix'] if c_type in DB else "U"
        self.counts[prefix] = self.counts.get(prefix, 0) + 1
        comp = Component(c_type, x, y, f"{prefix}{self.counts[prefix]}")
        comp.rotation = rotation
        self.components.append(comp)
        self.redraw_all()

    def set_mode(self, mode):
        self.mode = mode
        self.status.config(text=f"Tool: {mode.upper()}")
        self.selected_comps, self.selected_wires = [], []
        self.wire_start = None
        self.canvas.delete("ghost", "temp_wire")

    def rotate_command(self, e=None):
        if self.selected_comps:
            for c in self.selected_comps: c.rotate()
            self.redraw_all()
        elif self.mode not in ['select', 'wire', 'probe']: 
            self.ghost_rotation = (self.ghost_rotation+90)%360
            self.on_mouse_move(tk.Event())

    def delete_selection(self, e=None):
        for c in self.selected_comps: 
            if c in self.components: self.components.remove(c)
        for w in self.selected_wires: 
            if w in self.wires: self.wires.remove(w)
        self.selected_comps, self.selected_wires = [], []
        self.update_sidebar()
        self.redraw_all()

    def copy_selection(self, e=None):
        if not self.selected_comps: return
        self.clipboard = []
        rx, ry = self.selected_comps[0].x, self.selected_comps[0].y
        for c in self.selected_comps: 
            self.clipboard.append({'type': c.type, 'rx': c.x-rx, 'ry': c.y-ry, 'rot': c.rotation, 'p': copy.deepcopy(c.params)})

    def paste_selection(self, e=None):
        if not self.clipboard: return
        ref_x, ref_y = (self.selected_comps[0].x, self.selected_comps[0].y) if self.selected_comps else (0, 0)
        self.selected_comps = []
        for item in self.clipboard:
            prefix = DB[item['type']]['prefix']
            self.counts[prefix] = self.counts.get(prefix, 0) + 1
            new_c = Component(item['type'], ref_x+item['rx']+40, ref_y+item['ry']+40, f"{prefix}{self.counts[prefix]}")
            new_c.rotation = item['rot']; new_c.params = copy.deepcopy(item['p'])
            self.components.append(new_c); self.selected_comps.append(new_c)
        self.redraw_all()

    # ==========================================
    # VIEWPORT CONTROL
    # ==========================================

    def world_to_screen(self, wx, wy): return (wx * self.zoom) + self.offset_x, (wy * self.zoom) + self.offset_y
    def screen_to_world(self, sx, sy):
        wx = round(((sx - self.offset_x) / self.zoom) / GRID_SIZE) * GRID_SIZE
        wy = round(((sy - self.offset_y) / self.zoom) / GRID_SIZE) * GRID_SIZE
        return wx, wy
    def start_pan(self, e): self.pan_start = (e.x, e.y)
    def do_pan(self, e):
        if not hasattr(self, 'pan_start'): return
        self.offset_x += e.x - self.pan_start[0]; self.offset_y += e.y - self.pan_start[1]
        self.pan_start = (e.x, e.y)
        self.redraw_all()
    def do_zoom(self, e):
        wx, wy = self.screen_to_world(e.x, e.y)
        self.zoom *= 1.1 if e.delta > 0 else 0.9
        self.zoom = max(0.2, min(self.zoom, 5.0))
        self.offset_x, self.offset_y = e.x - wx*self.zoom, e.y - wy*self.zoom
        self.redraw_all()
    def on_mouse_move(self, event):
        wx, wy = self.screen_to_world(event.x, event.y)
        self._update_hover_state(wx, wy)
        if self.mode == 'wire' and self.wire_start: self._update_temp_wire_visual(wx, wy)
        elif self.mode not in ['select', 'wire', 'probe']: self._update_ghost(wx, wy)
    def _update_hover_state(self, wx, wy):
        self.hovered_pin = None
        for comp in self.components:
            for px, py in comp.get_pins():
                if abs(px - wx) < 5 and abs(py - wy) < 5: self.hovered_pin = (px, py); break
    def _update_temp_wire_visual(self, twx, twy):
        self.canvas.delete("temp_wire")
        if not self.wire_start: return
        sx, sy = self.wire_start
        if self.hovered_pin: twx, twy = self.hovered_pin
        p1, p2, p3 = self.world_to_screen(sx,sy), self.world_to_screen(twx,sy), self.world_to_screen(twx,twy)
        col = "green" if self.hovered_pin else "cyan"
        self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=col, dash=(2,2), tags="temp_wire")
        self.canvas.create_line(p2[0], p2[1], p3[0], p3[1], fill=col, dash=(2,2), tags="temp_wire")
    def _update_ghost(self, wx, wy):
        self.canvas.delete("ghost")
        sx, sy = self.world_to_screen(wx, wy)
        tk_img, _ = ComponentHelper.render_image(self.mode, self.ghost_rotation, self.zoom)
        if tk_img:
            self.ghost_img_ref = tk_img
            self.canvas.create_image(sx, sy, image=tk_img, anchor="center", tags="ghost")
    def on_right_click(self, e):
        self.canvas.focus_set()
        self.wire_start = None
        self.canvas.delete("temp_wire")
        if self.mode != 'select': self.set_mode('select')
        else: self.rotate_command()