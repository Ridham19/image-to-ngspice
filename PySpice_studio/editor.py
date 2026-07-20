import tkinter as tk
from tkinter import filedialog, messagebox, Menu
import copy
import subprocess
import os
import sys
import json
import re

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
        info = "SHORTCUTS:\n[W] Wire   [P] Probe\n[G] Ground\n[R] Resistor [C] Cap\n[L] Inductor [D] Diode\n[Del] Delete\n[Ctrl+C/V] Copy/Paste\n[Ctrl+R] Rotate"
        tk.Label(self.prop_frame, text=info, bg=COLOR_SIDEBAR_BG, fg="#888", justify="left", font=("Consolas", 8)).pack(side="bottom", pady=20)

    def _setup_shortcuts(self):
        keys = {'w': 'wire', 'p': 'probe', 'r': 'resistor', 'c': 'capacitor', 'l': 'inductor', 'd': 'diode', 'g': 'gnd'}
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
        
        type_mapping = {'voltage': 'source', 'ground': 'gnd', 'transistor': 'bjt_npn', 'mosfet': 'nmos'}

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
        
        try:
            log_path = os.path.join(cwd, "simulation.log")
            if os.path.exists(log_path):
                try: os.remove(log_path)
                except: pass
            
            # Start simulation with redirected output to simulation.log
            subprocess.Popen([self.ngspice_path, "-o", "simulation.log", filepath], cwd=cwd)
            
            # Check the log after 1 second
            self.root.after(1000, lambda: self.check_simulation_log(log_path, netlist_code))
        except Exception as e:
            messagebox.showerror("Execution Error", f"Failed to run simulation:\n{e}")

    def check_simulation_log(self, log_path, netlist):
        if not os.path.exists(log_path):
            return
        try:
            with open(log_path, "r") as f:
                log_content = f.read()
            logs = self.diagnose_simulation_output(log_content, netlist, self.components)
            # Show dialogue if there are errors or warnings
            has_issues = any(l['type'] in ['error', 'warning'] for l in logs)
            if has_issues:
                self.show_diagnostics_dialog(logs)
        except Exception as e:
            print("Error reading simulation log:", e)

    def diagnose_simulation_output(self, raw_output, netlist, components):
        logs = []
        lines = raw_output.split('\n') if raw_output else []
        netlist_lines = netlist.split('\n') if netlist else []

        def find_netlist_line(pattern, check_contains=True):
            for idx, line in enumerate(netlist_lines):
                if check_contains:
                    if pattern.lower() in line.lower():
                        return idx + 1, line
                else:
                    words = re.findall(r'\b\w+\b', line)
                    if any(w.lower() == pattern.lower() for w in words):
                        return idx + 1, line
            return None, None

        for line in lines:
            line_strip = line.strip()
            if not line_strip:
                continue

            log_type = None
            message = ""

            line_lower = line_strip.lower()
            if "fatal error" in line_lower:
                log_type = "error"
                message = line_strip
            elif "error" in line_lower or "singular matrix" in line_lower or "check node" in line_lower or "fail" in line_lower or "missing" in line_lower or "unknown device" in line_lower:
                log_type = "error"
                message = line_strip
            elif "warning" in line_lower:
                log_type = "warning"
                message = line_strip
            elif "note" in line_lower:
                log_type = "info"
                message = line_strip

            if log_type:
                words = re.findall(r'\b[\w_]+\b', message)
                matched_comp = None
                matched_node = None

                for comp in components:
                    name = comp.name if hasattr(comp, 'name') else comp.get('name', '')
                    if name and any(w.lower() == name.lower() for w in words):
                        matched_comp = name
                        break

                for w in words:
                    if w.isdigit() or w.lower().startswith("vsp_") or w.lower().startswith("vsn_") or w.lower() == "vsp" or w.lower() == "vsn":
                        matched_node = w

                netlist_line_num = None
                netlist_line_text = None

                if matched_comp:
                    netlist_line_num, netlist_line_text = find_netlist_line(matched_comp, check_contains=False)

                logs.append({
                    'type': log_type,
                    'message': message,
                    'component': matched_comp,
                    'node': matched_node,
                    'line': netlist_line_num,
                    'code': netlist_line_text
                })
        return logs

    def show_diagnostics_dialog(self, logs):
        diag_win = tk.Toplevel(self.root)
        diag_win.title("Simulation Diagnostics Console")
        diag_win.geometry("750x450")
        diag_win.configure(bg="#1E1E1E")

        title_lbl = tk.Label(diag_win, text="🔍 Simulation Diagnostics Console", bg="#1E1E1E", fg="#4CAF50", font=("Segoe UI", 12, "bold"))
        title_lbl.pack(pady=10)

        text_frame = tk.Frame(diag_win, bg="#1E1E1E")
        text_frame.pack(fill="both", expand=True, padx=15, pady=5)

        txt = tk.Text(text_frame, bg="#111111", fg="#E0E0E0", insertbackground="white", bd=0, highlightthickness=1, font=("Consolas", 10))
        txt.pack(fill="both", side="left", expand=True)

        scrollbar = tk.Scrollbar(text_frame, command=txt.yview)
        scrollbar.pack(side="right", fill="y")
        txt.config(yscrollcommand=scrollbar.set)

        for idx, log in enumerate(logs):
            prefix = "[INFO] "
            color = "#888888"
            if log['type'] == 'error':
                prefix = "[ERROR] ❌ "
                color = "#FF5252"
            elif log['type'] == 'warning':
                prefix = "[WARNING] ⚠️ "
                color = "#FFD740"

            txt.tag_config(f"tag_{idx}", foreground=color)
            txt.insert("end", prefix, f"tag_{idx}")
            txt.insert("end", log['message'] + "\n", f"tag_{idx}")
            
            details = []
            if log['component']:
                details.append(f"Component: {log['component']}")
            if log['node']:
                details.append(f"Node: {log['node']}")
            if log['line']:
                details.append(f"Netlist Line {log['line']}: '{log['code'].strip()}'")
            if details:
                txt.insert("end", "  └─ " + " | ".join(details) + "\n\n", "tag_details")

        txt.tag_config("tag_details", foreground="#88C0D0")
        txt.config(state="disabled")

        btn = tk.Button(diag_win, text="Dismiss Console", command=diag_win.destroy, bg="#0078D7", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=15, pady=5)
        btn.pack(pady=10)

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
            if key == 'custom_subckt':
                text_frame = tk.Frame(self.prop_container, bg=COLOR_SIDEBAR_BG)
                text_frame.pack(fill="x", pady=(0, 10))
                
                txt = tk.Text(text_frame, bg="#444", fg="white", insertbackground="white", bd=0, highlightthickness=1, height=8, width=25, font=("Consolas", 9))
                txt.insert("1.0", str(value))
                txt.pack(fill="both", side="left", expand=True)
                
                scrollbar = tk.Scrollbar(text_frame, command=txt.yview)
                scrollbar.pack(side="right", fill="y")
                txt.config(yscrollcommand=scrollbar.set)
                
                self.param_entries[key] = txt
            else:
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
            for key, entry in self.param_entries.items():
                if isinstance(entry, tk.Text):
                    comp.params[key] = entry.get("1.0", "end-1c")
                else:
                    comp.params[key] = entry.get()
            self.redraw_all()

    # ==========================================
    # TOOLBAR & CANVAS VISUALS
    # ==========================================

    def _init_professional_toolbar(self, parent):
        ribbon = tk.Frame(parent, bg=COLOR_TOOLBAR_BG) 
        ribbon.grid(row=0, column=0, sticky="ew")

        # Organize components by category
        categories = {}
        for c_type, data in DB.items():
            cat = data['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((c_type, data))

        # Render Dropdown Menus for Component Selection
        for cat in ['Passives', 'Active', 'Sources', 'Other']:
            if cat not in categories:
                continue
            mb = tk.Menubutton(ribbon, text=f"▾ {cat}", bg="#333333", fg=COLOR_TEXT_LIGHT, 
                               activebackground="#444444", activeforeground="white", 
                               relief="flat", font=("Segoe UI", 9), padx=12, pady=5)
            mb.pack(side="left", padx=3, pady=5)
            
            menu = tk.Menu(mb, tearoff=0, bg="#252526", fg=COLOR_TEXT_LIGHT, 
                           activebackground=COLOR_ACCENT_BLUE, activeforeground="white", bd=1, relief="solid")
            mb["menu"] = menu
            
            for c_type, data in categories[cat]:
                lbl = data['label']
                menu.add_command(label=lbl, command=lambda t=c_type: self.set_mode(t))

        # Spacer between dropdowns and drawing tools
        tk.Frame(ribbon, width=15, bg=COLOR_TOOLBAR_BG).pack(side="left")

        # Drawing Tools (Direct Buttons)
        g_tools = tk.Frame(ribbon, bg=COLOR_TOOLBAR_BG)
        g_tools.pack(side="left", pady=5)
        tk.Button(g_tools, text="✏️ WIRE", command=lambda: self.set_mode('wire'), 
                  bg=COLOR_ACCENT_BLUE, fg="white", relief="flat", font=("Segoe UI", 9), padx=10, pady=3).pack(side="left", padx=2)
        tk.Button(g_tools, text="🔍 PROBE", command=lambda: self.set_mode("probe"), 
                  bg="#FF9800", fg="white", relief="flat", font=("Segoe UI", 9), padx=10, pady=3).pack(side="left", padx=2)
        tk.Button(g_tools, text="⏚ GND", command=lambda: self.set_mode('gnd'), 
                  bg="#43A047", fg="white", relief="flat", font=("Segoe UI", 9), padx=10, pady=3).pack(side="left", padx=2)

        # Spacer between drawing tools and simulation control
        tk.Frame(ribbon, width=15, bg=COLOR_TOOLBAR_BG).pack(side="left")

        # Simulation Control (Direct Buttons)
        g_sim = tk.Frame(ribbon, bg=COLOR_TOOLBAR_BG)
        g_sim.pack(side="left", pady=5)
        tk.Button(g_sim, text="Config", command=self.open_sim_dialog, 
                  bg="#555555", fg="white", relief="flat", font=("Segoe UI", 9), padx=10, pady=3).pack(side="left", padx=2)
        tk.Button(g_sim, text="RUN", command=self.run_simulation, 
                  bg="#2E7D32", fg="white", font=("Segoe UI", 9, "bold"), relief="flat", padx=15, pady=3).pack(side="left", padx=5)

        self.status = tk.Label(ribbon, text="Ready", bg=COLOR_TOOLBAR_BG, fg="#888888", font=("Segoe UI", 9))
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
            if sx != wx:
                self.wires.append(((sx, sy), (wx, sy)))
                self._check_and_add_junction(sx, sy)
                self._check_and_add_junction(wx, sy)
            if sy != wy:
                self.wires.append(((wx, sy), (wx, wy)))
                self._check_and_add_junction(wx, sy)
                self._check_and_add_junction(wx, wy)
            self.redraw_all()
            self.wire_start = None if self.hovered_pin else (wx, wy)
        else:
            self.wire_start = (wx, wy)

    def _handle_selection_click(self, wx, wy):
        clicked_comp = next((c for c in self.components if abs(c.x-wx)<30 and abs(c.y-wy)<30), None)
        if clicked_comp:
            self.selected_comps, self.selected_wires = [clicked_comp], []
            self.drag_start_world = (wx, wy)
            self.drag_start_positions = {c: (c.x, c.y) for c in self.selected_comps}
            self.pre_existing_overlap_pairs = self._get_overlap_pairs()
            self.pre_existing_wire_overlaps = self._get_wire_overlaps()
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
        elif self.mode == 'select' and self.selected_comps and self.drag_start_world:
            collision = False
            for c in self.selected_comps:
                for other in self.components:
                    if other == c or other in self.selected_comps:
                        continue
                    if c.type in ['label', 'junction', 'crossover', 'terminal', 'gnd'] or other.type in ['label', 'junction', 'crossover', 'terminal', 'gnd']:
                        continue
                    if self.components_overlap(c, other):
                        pair = tuple(sorted([c.name, other.name]))
                        if not hasattr(self, 'pre_existing_overlap_pairs') or pair not in self.pre_existing_overlap_pairs:
                            collision = True
                            break
                if collision:
                    break
                
                for w in self.wires:
                    if self.does_comp_overlap_wire(c, w):
                        overlap = (c.name, w)
                        if not hasattr(self, 'pre_existing_wire_overlaps') or overlap not in self.pre_existing_wire_overlaps:
                            collision = True
                            break
                if collision:
                    break
            
            if collision:
                for c, pos in self.drag_start_positions.items():
                    c.x, c.y = pos
                self.status.config(text="⚠️ Overlap detected! Rollback snapped.")
                self.redraw_all()
            else:
                self.status.config(text="Ready")
            
            self.drag_start_positions = {}
            self.pre_existing_overlap_pairs = set()
            self.pre_existing_wire_overlaps = set()

        self.drag_start_world = None

    # ==========================================
    # DRAWING UTILITIES
    # ==========================================

    def get_hitbox(self, comp):
        w, h = 80, 40  # default
        if comp.type in DB:
            shape = DB[comp.type]['shape']
            if shape == 'label':
                w, h = 10, 10
            elif shape in ['3_pin_bjt', '3_pin_fet']:
                w, h = 40, 80
            elif shape == 'v_source':
                w, h = 50, 80
            elif shape == '1_pin':
                w, h = 40, 30
            elif shape == 'opamp':
                w, h = 70, 60
            elif shape == 'ic':
                num_pins = int(comp.params.get('num_pins', '2'))
                half = (num_pins + 1) // 2
                body_h = max(30, half * 20)
                w, h = 100, body_h + 20
            elif shape == 'transformer':
                w, h = 80, 60
        if comp.rotation in [90, 270]:
            w, h = h, w
        return w, h

    def components_overlap(self, c1, c2):
        w1, h1 = self.get_hitbox(c1)
        w2, h2 = self.get_hitbox(c2)
        return (
            abs(c1.x - c2.x) < (w1 + w2) / 2 and
            abs(c1.y - c2.y) < (h1 + h2) / 2
        )

    def does_comp_overlap_wire(self, comp, wire):
        w, h = self.get_hitbox(comp)
        bx1, by1 = comp.x - w/2, comp.y - h/2
        bx2, by2 = comp.x + w/2, comp.y + h/2
        (x1, y1), (x2, y2) = wire
        
        if x1 == x2: # Vertical wire
            if bx1 <= x1 <= bx2:
                if min(y1, y2) < by2 and max(y1, y2) > by1:
                    return True
        elif y1 == y2: # Horizontal wire
            if by1 <= y1 <= by2:
                if min(x1, x2) < bx2 and max(x1, x2) > bx1:
                    return True
        return False

    def _get_overlap_pairs(self):
        pairs = set()
        for i, c1 in enumerate(self.components):
            for j in range(i + 1, len(self.components)):
                c2 = self.components[j]
                if c1.type in ['label', 'junction', 'crossover', 'terminal', 'gnd'] or c2.type in ['label', 'junction', 'crossover', 'terminal', 'gnd']:
                    continue
                if self.components_overlap(c1, c2):
                    pairs.add(tuple(sorted([c1.name, c2.name])))
        return pairs

    def _get_wire_overlaps(self):
        overlaps = set()
        for c in self.components:
            if c.type in ['label', 'junction', 'crossover', 'terminal', 'gnd']:
                continue
            for w in self.wires:
                if self.does_comp_overlap_wire(c, w):
                    overlaps.add((c.name, w))
        return overlaps

    def _check_and_add_junction(self, x, y):
        x = round(x / GRID_SIZE) * GRID_SIZE
        y = round(y / GRID_SIZE) * GRID_SIZE
        for c in self.components:
            if c.x == x and c.y == y:
                return
        for w in self.wires:
            (x1, y1), (x2, y2) = w
            if x1 == x2:
                if x == x1 and min(y1, y2) < y < max(y1, y2):
                    self.create_junction(x, y)
                    break
            elif y1 == y2:
                if y == y1 and min(x1, x2) < x < max(x1, x2):
                    self.create_junction(x, y)
                    break

    def create_junction(self, x, y):
        self.counts['J'] = self.counts.get('J', 0) + 1
        comp = Component('junction', x, y, f"J{self.counts['J']}")
        self.components.append(comp)

    def cleanup_junctions(self):
        junctions = [c for c in self.components if c.type == 'junction']
        for j in junctions:
            degree = 0
            for w in self.wires:
                (x1, y1), (x2, y2) = w
                if (x1 == j.x and y1 == j.y) or (x2 == j.x and y2 == j.y):
                    degree += 1
                elif x1 == x2 and x1 == j.x and min(y1, y2) < j.y < max(y1, y2):
                    degree += 2
                elif y1 == y2 and y1 == j.y and min(x1, x2) < j.x < max(x1, x2):
                    degree += 2
            
            if degree < 3:
                if j in self.components:
                    self.components.remove(j)

    def redraw_all(self):
        self.cleanup_junctions()
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
        tk_img, _ = ComponentHelper.render_image(comp.type, comp.rotation, self.zoom, comp)
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