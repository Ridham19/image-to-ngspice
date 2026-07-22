import tkinter as tk
from tkinter import filedialog, messagebox, Menu, simpledialog
import copy
import subprocess
import os
import sys
import json
import re
import heapq

# --- AI PIPELINE INTEGRATION ---
# Ensure PySpice_studio directory is in sys.path for internal packages (core, wire_tracer)
studio_dir = os.path.dirname(__file__)
if studio_dir not in sys.path:
    sys.path.insert(0, studio_dir)

ai_path = os.path.abspath(os.path.join(studio_dir, '..', 'proper'))
if ai_path not in sys.path:
    sys.path.append(ai_path)

try:
    from modules.model import ComponentDetector  # type: ignore
    from modules.processing import preprocess_image, separate_layers  # type: ignore
    from modules.netlist import trace_nodes  # type: ignore
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

class InteractiveAIDebugDialog:
    def __init__(self, parent, bgr_image, detected_comps, initial_connections, pin_anchors):
        import cv2
        from PIL import Image, ImageTk

        self.parent = parent
        self.bgr_image = bgr_image
        self.detected_comps = copy.deepcopy(detected_comps)
        self.initial_connections = copy.deepcopy(initial_connections)
        self.connections = copy.deepcopy(initial_connections)
        self.pin_anchors = copy.deepcopy(pin_anchors)
        
        self.junction_counter = 0
        self.tool = 'select'  # 'select', 'connect', 'junction', 'delete'
        self.pending_pin = None
        self.hovered_pin = None
        self.hovered_conn_idx = None
        self.hovered_comp = None
        
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.is_dragging = False
        self.mouse_x = 0
        self.mouse_y = 0
        
        self.result_accepted = False

        # Convert OpenCV BGR to PIL Image
        rgb_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        self.orig_pil_img = Image.fromarray(rgb_img)
        self.img_w, self.img_h = self.orig_pil_img.size

        # Create TopLevel Window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Interactive AI Vision Debugger & Connection Editor")
        self.dialog.configure(bg="#1E1E1E")
        self.dialog.geometry("1100x800")

        self._setup_ui()
        self._fit_to_screen()
        self.redraw()
        
        self.dialog.transient(parent)
        self.dialog.grab_set()

    def _setup_ui(self):
        # 1. Top Toolbar
        tb = tk.Frame(self.dialog, bg="#2D2D2D")
        tb.pack(side=tk.TOP, fill=tk.X)

        self.btn_select = tk.Button(tb, text="👆 Select / Edit Value", command=lambda: self.set_tool('select'),
                                    bg="#0078D7", fg="white", font=("Segoe UI", 10, "bold"), relief=tk.RAISED)
        self.btn_select.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_connect = tk.Button(tb, text="🔌 Connect Pins", command=lambda: self.set_tool('connect'),
                                     bg="#333333", fg="white", font=("Segoe UI", 10), relief=tk.RAISED)
        self.btn_connect.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_junction = tk.Button(tb, text="❖ Add Junction", command=lambda: self.set_tool('junction'),
                                      bg="#333333", fg="white", font=("Segoe UI", 10), relief=tk.RAISED)
        self.btn_junction.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_delete = tk.Button(tb, text="🗑️ Delete Wire/Junction", command=lambda: self.set_tool('delete'),
                                    bg="#333333", fg="white", font=("Segoe UI", 10), relief=tk.RAISED)
        self.btn_delete.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_reset = tk.Button(tb, text="🔄 Reset Auto-Trace", command=self.reset_connections,
                                   bg="#444444", fg="white", font=("Segoe UI", 10), relief=tk.RAISED)
        self.btn_reset.pack(side=tk.LEFT, padx=15, pady=5)

        # 2. Main Canvas
        self.canvas_frame = tk.Frame(self.dialog, bg="#181818")
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#181818", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Canvas Events
        self.canvas.bind("<ButtonPress-1>", self.on_left_click)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(1.1, e.x, e.y))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(0.9, e.x, e.y))

        # 3. Bottom Bar
        bottom_bar = tk.Frame(self.dialog, bg="#252526", height=50)
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_stats = tk.Label(bottom_bar, text="", bg="#252526", fg="#00E5FF", font=("Segoe UI", 10, "bold"))
        self.lbl_stats.pack(side=tk.LEFT, padx=15, pady=10)

        self.lbl_info = tk.Label(bottom_bar, text="Click a component to edit value, or select a tool to edit wires.",
                                 bg="#252526", fg="#CCCCCC", font=("Segoe UI", 10))
        self.lbl_info.pack(side=tk.LEFT, padx=15, pady=10)

        btn_accept = tk.Button(bottom_bar, text="✓ Accept & Load Circuit", command=self.accept,
                               bg="#10B981", fg="white", font=("Segoe UI", 11, "bold"), padx=15, pady=5)
        btn_accept.pack(side=tk.RIGHT, padx=15, pady=8)

        btn_cancel = tk.Button(bottom_bar, text="Cancel", command=self.dialog.destroy,
                               bg="#444444", fg="white", font=("Segoe UI", 10), padx=10, pady=5)
        btn_cancel.pack(side=tk.RIGHT, padx=5, pady=8)

    def set_tool(self, mode):
        self.tool = mode
        self.pending_pin = None
        for b, m in [(self.btn_select, 'select'), (self.btn_connect, 'connect'),
                     (self.btn_junction, 'junction'), (self.btn_delete, 'delete')]:
            if m == mode:
                b.configure(bg="#0078D7", font=("Segoe UI", 10, "bold"))
            else:
                b.configure(bg="#333333", font=("Segoe UI", 10))
        
        info_msgs = {
            'select': 'Select Mode: Click component box to edit value, or drag canvas to pan.',
            'connect': 'Connect Mode: Click Pin 1, then click Pin 2 to connect.',
            'junction': 'Junction Mode: Click image background to add a Junction point.',
            'delete': 'Delete Mode: Click a wire line or junction diamond to remove it.'
        }
        self.lbl_info.config(text=info_msgs.get(mode, ""))
        self.redraw()

    def reset_connections(self):
        self.connections = copy.deepcopy(self.initial_connections)
        self.pending_pin = None
        self.lbl_info.config(text="🔄 Reset connections to auto-traced state.")
        self.update_stats()
        self.redraw()

    def _fit_to_screen(self):
        self.dialog.update_idletasks()
        cw = self.canvas.winfo_width() or 1000
        ch = self.canvas.winfo_height() or 650
        scale = min((cw - 40) / self.img_w, (ch - 40) / self.img_h)
        self.zoom = max(0.2, min(scale, 2.0))
        self.pan_x = (cw - self.img_w * self.zoom) / 2
        self.pan_y = (ch - self.img_h * self.zoom) / 2

    def img_to_canvas(self, x, y):
        return x * self.zoom + self.pan_x, y * self.zoom + self.pan_y

    def canvas_to_img(self, cx, cy):
        return (cx - self.pan_x) / self.zoom, (cy - self.pan_y) / self.zoom

    def update_stats(self):
        comp_count = len([c for c in self.detected_comps if c.get('type') not in ['wire', 'junction', 'text']])
        conn_count = len(self.connections)
        pin_count = len(self.pin_anchors)
        self.lbl_stats.config(text=f"📊 Components: {comp_count} | Wires: {conn_count} | Pins: {pin_count}")

    def redraw(self):
        from PIL import Image, ImageTk
        self.canvas.delete("all")
        self.update_stats()

        # 1. Render scaled background image
        scaled_w = int(self.img_w * self.zoom)
        scaled_h = int(self.img_h * self.zoom)
        if scaled_w > 0 and scaled_h > 0:
            resized_pil = self.orig_pil_img.resize((scaled_w, scaled_h), Image.Resampling.BILINEAR)
            self.tk_img = ImageTk.PhotoImage(resized_pil)
            self.canvas.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_img)

        # 2. Draw component bounding boxes
        for comp in self.detected_comps:
            box = comp.get('box')
            if not box or len(box) < 4: continue
            x1, y1 = self.img_to_canvas(box[0], box[1])
            x2, y2 = self.img_to_canvas(box[0] + box[2], box[1] + box[3])
            
            is_hovered = (self.hovered_comp == comp)
            outline_color = "#10B981" if is_hovered else "#00FFCD"
            width = 3 if is_hovered else 2
            
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=outline_color, width=width)
            
            lbl = comp.get('name', comp.get('type', ''))
            val = comp.get('value', '')
            if val and val != "TEXT_FOUND":
                lbl += f" ({val})"
            
            self.canvas.create_text(x1 + 4, y1 - 8, text=lbl, fill="#00FFCD" if is_hovered else "#FFFFFF",
                                    anchor=tk.SW, font=("Segoe UI", 10, "bold"))

        # 3. Draw connections
        for idx, conn in enumerate(self.connections):
            p1_ref = conn['pin1']
            p2_ref = conn['pin2']
            p1 = self.find_pin(p1_ref['comp_idx'], p1_ref['pin_id'])
            p2 = self.find_pin(p2_ref['comp_idx'], p2_ref['pin_id'])
            if not p1 or not p2: continue

            cx1, cy1 = self.img_to_canvas(p1['x'], p1['y'])
            cx2, cy2 = self.img_to_canvas(p2['x'], p2['y'])

            is_hovered = (self.hovered_conn_idx == idx)
            line_color = "#EF4444" if is_hovered else "#00E5FF"
            line_width = 4 if is_hovered else 2.5
            dash = (4, 4) if is_hovered else None

            self.canvas.create_line(cx1, cy1, cx2, cy2, fill=line_color, width=line_width, dash=dash)

            if is_hovered and self.tool == 'delete':
                mx, my = (cx1 + cx2) / 2, (cy1 + cy2) / 2
                self.canvas.create_oval(mx - 10, my - 10, mx + 10, my + 10, fill="#EF4444", outline="white", width=2)
                self.canvas.create_line(mx - 5, my - 5, mx + 5, my + 5, fill="white", width=2)
                self.canvas.create_line(mx + 5, my - 5, mx - 5, my + 5, fill="white", width=2)

        # 4. Draw pin anchors
        r = max(6, int(8 * self.zoom))
        for pin in self.pin_anchors:
            cx, cy = self.img_to_canvas(pin['x'], pin['y'])
            is_junc = pin.get('isJunction', False) or str(pin.get('comp_idx')) == '-1'
            
            is_hovered = (self.hovered_pin and str(self.hovered_pin['comp_idx']) == str(pin['comp_idx']) and str(self.hovered_pin['pin_id']) == str(pin['pin_id']))
            is_pending = (self.pending_pin and str(self.pending_pin['comp_idx']) == str(pin['comp_idx']) and str(self.pending_pin['pin_id']) == str(pin['pin_id']))

            if is_junc:
                # Diamond
                points = [cx, cy - r*1.2, cx + r*1.2, cy, cx, cy + r*1.2, cx - r*1.2, cy]
                fill_color = "#00E5FF" if is_pending else ("#EF4444" if is_hovered and self.tool == 'delete' else "#FFC832")
                self.canvas.create_polygon(points, fill=fill_color, outline="white", width=2)
                self.canvas.create_text(cx + r + 6, cy, text=f"J{pin['pin_id']}", fill="#FFC832", anchor=tk.W, font=("Segoe UI", 9, "bold"))
            else:
                # Circle
                fill_color = "#00E5FF" if is_pending else ("#10B981" if is_hovered else "#777777")
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=fill_color, outline="white", width=2 if is_hovered else 1)
                self.canvas.create_text(cx + r + 4, cy, text=f"{pin['comp_idx']}_{pin['pin_id']}", fill="white", anchor=tk.W, font=("Segoe UI", 9))

        # 5. Draw pending wire
        if self.pending_pin and self.tool == 'connect':
            p1 = self.find_pin(self.pending_pin['comp_idx'], self.pending_pin['pin_id'])
            if p1:
                cx1, cy1 = self.img_to_canvas(p1['x'], p1['y'])
                self.canvas.create_line(cx1, cy1, self.mouse_x, self.mouse_y, fill="#00E5FF", width=2, dash=(6, 4))

    def find_pin(self, comp_idx, pin_id):
        c1 = str(comp_idx).replace("comp_", "")
        p1 = str(pin_id).replace("pin_", "")
        for p in self.pin_anchors:
            c2 = str(p.get('comp_idx', '')).replace("comp_", "")
            p2 = str(p.get('pin_id', '')).replace("pin_", "")
            if c1 == c2 and p1 == p2:
                return p
        return None

    def hit_test_pin(self, img_x, img_y, threshold=20):
        for pin in self.pin_anchors:
            dist = ((pin['x'] - img_x)**2 + (pin['y'] - img_y)**2)**0.5
            if dist < (threshold / self.zoom):
                return pin
        return None

    def hit_test_comp(self, img_x, img_y):
        for comp in self.detected_comps:
            box = comp.get('box')
            if not box or len(box) < 4: continue
            if box[0] <= img_x <= box[0] + box[2] and box[1] <= img_y <= box[1] + box[3]:
                return comp
        return None

    def hit_test_connection(self, img_x, img_y, threshold=15):
        for idx, conn in enumerate(self.connections):
            p1 = self.find_pin(conn['pin1']['comp_idx'], conn['pin1']['pin_id'])
            p2 = self.find_pin(conn['pin2']['comp_idx'], conn['pin2']['pin_id'])
            if not p1 or not p2: continue

            x1, y1 = p1['x'], p1['y']
            x2, y2 = p2['x'], p2['y']
            dx, dy = x2 - x1, y2 - y1
            if dx == 0 and dy == 0: continue
            t = max(0, min(1, ((img_x - x1) * dx + (img_y - y1) * dy) / (dx*dx + dy*dy)))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist = ((img_x - proj_x)**2 + (img_y - proj_y)**2)**0.5
            if dist < (threshold / self.zoom):
                return idx
        return None

    def on_left_click(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.is_dragging = False

        img_x, img_y = self.canvas_to_img(event.x, event.y)
        clicked_pin = self.hit_test_pin(img_x, img_y)
        clicked_comp = self.hit_test_comp(img_x, img_y)

        if self.tool == 'select':
            if clicked_comp:
                old_val = clicked_comp.get('value', '')
                if old_val == "TEXT_FOUND": old_val = ""
                new_val = simpledialog.askstring("Edit Component Value", f"Enter value for {clicked_comp.get('name')} ({clicked_comp.get('type')}):", initialvalue=old_val, parent=self.dialog)
                if new_val is not None:
                    clicked_comp['value'] = new_val
                    self.lbl_info.config(text=f"✍️ Updated {clicked_comp.get('name')} value to {new_val}")
                    self.redraw()
        elif self.tool == 'connect':
            if clicked_pin:
                if not self.pending_pin:
                    self.pending_pin = {'comp_idx': clicked_pin['comp_idx'], 'pin_id': clicked_pin['pin_id']}
                    self.lbl_info.config(text=f"Selected pin {clicked_pin['comp_idx']}_{clicked_pin['pin_id']}. Now click target pin.")
                else:
                    p1 = self.pending_pin
                    p2 = {'comp_idx': clicked_pin['comp_idx'], 'pin_id': clicked_pin['pin_id']}
                    if str(p1['comp_idx']) == str(p2['comp_idx']) and str(p1['pin_id']) == str(p2['pin_id']):
                        self.pending_pin = None
                        self.lbl_info.config(text="Same pin — cancelled connection.")
                    else:
                        self.connections.append({'pin1': p1, 'pin2': p2})
                        self.lbl_info.config(text=f"✅ Connected {p1['comp_idx']}_{p1['pin_id']} → {p2['comp_idx']}_{p2['pin_id']}")
                        self.pending_pin = None
                self.redraw()
        elif self.tool == 'junction':
            j_id = str(self.junction_counter)
            self.junction_counter += 1
            new_junc = {'comp_idx': -1, 'pin_id': j_id, 'x': int(img_x), 'y': int(img_y), 'isJunction': True}
            self.pin_anchors.append(new_junc)
            self.detected_comps.append({'type': 'junction', 'name': f"J{j_id}", 'center': [int(img_x), int(img_y)], 'box': [int(img_x)-10, int(img_y)-10, 20, 20]})
            self.lbl_info.config(text=f"❖ Added Junction J{j_id}")
            self.redraw()
        elif self.tool == 'delete':
            if clicked_pin and (clicked_pin.get('isJunction') or str(clicked_pin['comp_idx']) == '-1'):
                j_id = clicked_pin['pin_id']
                self.pin_anchors = [p for p in self.pin_anchors if not (str(p['comp_idx']) == '-1' and str(p['pin_id']) == str(j_id))]
                self.connections = [c for c in self.connections if not (
                    (str(c['pin1']['comp_idx']) == '-1' and str(c['pin1']['pin_id']) == str(j_id)) or
                    (str(c['pin2']['comp_idx']) == '-1' and str(c['pin2']['pin_id']) == str(j_id))
                )]
                self.lbl_info.config(text=f"🗑️ Deleted Junction J{j_id}")
                self.redraw()
            else:
                conn_idx = self.hit_test_connection(img_x, img_y)
                if conn_idx is not None:
                    removed = self.connections.pop(conn_idx)
                    self.lbl_info.config(text=f"🗑️ Removed connection {removed['pin1']['comp_idx']}_{removed['pin1']['pin_id']} → {removed['pin2']['comp_idx']}_{removed['pin2']['pin_id']}")
                    self.redraw()

    def on_mouse_drag(self, event):
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        if abs(dx) > 3 or abs(dy) > 3:
            self.is_dragging = True
            self.pan_x += dx
            self.pan_y += dy
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.redraw()

    def on_left_release(self, event):
        self.is_dragging = False

    def on_mouse_move(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y
        img_x, img_y = self.canvas_to_img(event.x, event.y)

        self.hovered_pin = self.hit_test_pin(img_x, img_y)
        self.hovered_comp = self.hit_test_comp(img_x, img_y)
        if self.tool == 'delete':
            self.hovered_conn_idx = self.hit_test_connection(img_x, img_y)
        else:
            self.hovered_conn_idx = None

        self.redraw()

    def on_mouse_wheel(self, event):
        factor = 1.15 if event.delta > 0 else 0.85
        self._zoom_at(factor, event.x, event.y)

    def _zoom_at(self, factor, cx, cy):
        old_zoom = self.zoom
        new_zoom = max(0.15, min(old_zoom * factor, 5.0))
        img_x, img_y = self.canvas_to_img(cx, cy)
        self.zoom = new_zoom
        self.pan_x = cx - img_x * self.zoom
        self.pan_y = cy - img_y * self.zoom
        self.redraw()

    def accept(self):
        self.result_accepted = True
        self.dialog.destroy()

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
        self.undo_stack = []
        self.redo_stack = []
        
        self.selection_box_start = None
        self.drag_start_world = None
        self.wire_start = None
        self.ghost_rotation = 0
        self.hovered_pin = None
        self.drag_attached_wires = {}        # {wire_idx: {'move_p1': bool, 'move_p2': bool}}
        self.drag_start_wire_positions = {}  # {wire_idx: (p1, p2)} snapshot for rollback

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

        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        
        file_menu.add_command(label="📸 Import from Image (AI)...", command=self.import_from_image)
        file_menu.add_command(label="📄 Import AI JSON...", command=self.import_from_ai)
        file_menu.add_separator()
        file_menu.add_command(label="🖼️ Export Image (PNG)...", command=self.export_schematic_image)
        file_menu.add_command(label="💾 Export Netlist (.cir)...", command=self.export_netlist_file)
        file_menu.add_separator()
        file_menu.add_command(label="Settings (Set ngspice path)", command=self.open_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Edit Menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="↶ Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="↷ Redo", command=self.redo, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Select All", command=self.select_all, accelerator="Ctrl+A")
        edit_menu.add_command(label="Duplicate", command=self.duplicate_selection, accelerator="Ctrl+D")
        edit_menu.add_command(label="Rotate", command=self.rotate_command, accelerator="Ctrl+R")
        edit_menu.add_command(label="Delete", command=self.delete_selection, accelerator="Delete")
        edit_menu.add_separator()
        edit_menu.add_command(label="↺ Reroute All Wires", command=self.reroute_all_wires)

        # View Menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="🔍 Fit to Screen", command=self.fit_to_screen, accelerator="F")

        # Help Menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="⌨️ Keyboard Shortcuts Reference", command=self.show_shortcuts_dialog, accelerator="F1")

    def _add_sidebar_header(self, text):
        tk.Label(self.prop_frame, text=text, bg="#333", fg=COLOR_TEXT_LIGHT, font=("Segoe UI", 9, "bold")).pack(fill="x", pady=(10, 0))

    def _add_sidebar_footer_shortcuts(self):
        info = "SHORTCUTS:\n[W] Wire [P] Probe [G] GND\n[J] Jxn  [R] Res   [C] Cap\n[L] Ind  [D] Diode [F] Fit\n[Ctrl+Z/Y] Undo/Redo\n[Ctrl+A/D] All/Dup\n[Ctrl+C/V] Copy/Paste\n[Del] Delete  [F1] Help"
        tk.Label(self.prop_frame, text=info, bg=COLOR_SIDEBAR_BG, fg="#888", justify="left", font=("Consolas", 8)).pack(side="bottom", pady=15)

    def _setup_shortcuts(self):
        keys = {'w': 'wire', 'p': 'probe', 'r': 'resistor', 'c': 'capacitor',
                'l': 'inductor', 'd': 'diode', 'g': 'gnd', 'j': 'junction'}
        for key, mode in keys.items(): self.root.bind(key, lambda e, m=mode: self.set_mode(m))
        self.root.bind('<Delete>', self.delete_selection)
        self.root.bind('<Control-r>', self.rotate_command)
        self.root.bind('<Control-c>', self.copy_selection)
        self.root.bind('<Control-v>', self.paste_selection)
        self.root.bind('<Control-z>', self.undo)
        self.root.bind('<Control-y>', self.redo)
        self.root.bind('<Control-Z>', self.redo)
        self.root.bind('<Control-a>', self.select_all)
        self.root.bind('<Control-d>', self.duplicate_selection)
        self.root.bind('<Escape>', self.cancel_current_action)
        self.root.bind('<f>', lambda e: self.fit_to_screen())
        self.root.bind('<F1>', lambda e: self.show_shortcuts_dialog())

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

            import cv2
            from PIL import Image, ImageTk
            import base64

            # 1. Read image and run YOLO Object Detection
            bgr_image = cv2.imread(img_path)
            if bgr_image is None:
                messagebox.showerror("Image Error", "Failed to decode the selected image file.")
                self.status.config(text="Ready")
                return

            detector = ComponentDetector()
            json_output_path = os.path.join(os.path.dirname(__file__), "detected_components.json")
            detected_comps = detector.detect(bgr_image, output_file=json_output_path)
            
            # 2. Run Wire Tracing Pipeline
            from core.processing import compute_pin_anchors, TRANSPARENT_TYPES, preprocess_image as core_preprocess  # type: ignore
            from wire_tracer.tracer import trace_wires  # type: ignore
            from wire_tracer.config import WireTracerConfig  # type: ignore
            from wire_tracer.utils import draw_debug_overlay  # type: ignore
            
            _, _, core_binary = core_preprocess(bgr_image)
            pin_anchors = compute_pin_anchors(detected_comps, wire_mask=core_binary)
            
            wt_components = []
            for idx, comp in enumerate(detected_comps):
                ctype = comp.get("type", "")
                if ctype in TRANSPARENT_TYPES:
                    continue
                
                wt_comp = {
                    "id": f"comp_{idx}",
                    "label": ctype,
                    "bbox": comp.get("box", [0,0,0,0]),
                    "pins": []
                }
                
                for anchor in pin_anchors:
                    if anchor["comp_idx"] == idx:
                        wt_comp["pins"].append({
                            "id": f"{idx}_{anchor['pin_id']}",
                            "loc": [anchor["x"], anchor["y"]]
                        })
                
                wt_components.append(wt_comp)
                
            config = WireTracerConfig(
                pin_pad=8,
                max_search_radius=45,
                enhance_wire_continuity=True,
                h_wire_kernel_len=40,
                v_wire_kernel_len=40
            )

            netlist, debug_info = trace_wires(
                bgr_image, 
                wt_components,
                config=config, 
                method="connected_components", 
                debug=True
            )
            
            connections = []
            for net_pins in netlist:
                for i in range(len(net_pins) - 1):
                    try:
                        p1_parts = net_pins[i].rsplit("_", 1)
                        p2_parts = net_pins[i+1].rsplit("_", 1)
                        p1_idx = int(p1_parts[0]) if p1_parts[0].lstrip('-').isdigit() else p1_parts[0]
                        p1_pid = int(p1_parts[1]) if p1_parts[1].isdigit() else p1_parts[1]
                        p2_idx = int(p2_parts[0]) if p2_parts[0].lstrip('-').isdigit() else p2_parts[0]
                        p2_pid = int(p2_parts[1]) if p2_parts[1].isdigit() else p2_parts[1]
                        connections.append({
                            "pin1": {"comp_idx": p1_idx, "pin_id": p1_pid},
                            "pin2": {"comp_idx": p2_idx, "pin_id": p2_pid}
                        })
                    except Exception as err:
                        print(f"Skipping malformed net pin pair ({net_pins[i]}, {net_pins[i+1]}): {err}")
            
            # 3. Launch Interactive AI Vision Debugger & Connection Editor Modal
            self.status.config(text="Review & Edit connections in interactive window...")
            self.root.update()
            
            dlg = InteractiveAIDebugDialog(self.root, bgr_image, detected_comps, connections, pin_anchors)
            self.root.wait_window(dlg.dialog)
            
            if dlg.result_accepted:
                # 4. Load Data & User-Verified Connections to Canvas
                self._load_ai_data_to_canvas(dlg.detected_comps, connections=dlg.connections, pin_anchors=dlg.pin_anchors)
                self.status.config(text="Ready")
                messagebox.showinfo("AI Import Success", f"AI placed components and loaded interactive connections!")
            else:
                self.status.config(text="Ready")

        except Exception as e:
            import traceback
            traceback.print_exc()
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

    def _does_segment_intersect_component(self, pA, pB, components, exclude_comps=None):
        if exclude_comps is None: exclude_comps = []
        ignore_types = {'label', 'junction', 'crossover', 'terminal'}
        for c in components:
            if c in exclude_comps or getattr(c, 'type', '') in ignore_types:
                continue
            w, h = 50, 50
            if getattr(c, 'rotation', 0) in (90, 270):
                w, h = h, w
            margin = 2
            xmin, xmax = c.x - w / 2 + margin, c.x + w / 2 - margin
            ymin, ymax = c.y - h / 2 + margin, c.y + h / 2 - margin

            if abs(pA[1] - pB[1]) < 1:  # Horizontal segment
                y = pA[1]
                min_x, max_x = min(pA[0], pB[0]), max(pA[0], pB[0])
                if ymin < y < ymax and min_x < xmax and max_x > xmin:
                    return True
            elif abs(pA[0] - pB[0]) < 1:  # Vertical segment
                x = pA[0]
                min_y, max_y = min(pA[1], pB[1]), max(pA[1], pB[1])
                if xmin < x < xmax and min_y < ymax and max_y > ymin:
                    return True
        return False

    def _load_ai_data_to_canvas(self, ai_data, connections=None, pin_anchors=None):
        """Translates AI JSON array into Canvas Components AND Mathematically Routes Wires."""
        self.components = []
        self.wires = []
        self.selected_comps = []
        self.selected_wires = []
        
        type_mapping = {'voltage': 'source', 'ground': 'gnd', 'transistor': 'bjt_npn', 'mosfet': 'nmos'}
        NON_COMPONENT_TYPES = ['wire', 'junction', 'crossover', 'terminal', 'text']

        # 1. Dynamically calculate SCALE_FACTOR based on average component width
        sum_w, count_w = 0, 0
        for item in ai_data:
            if item.get('type') not in NON_COMPONENT_TYPES and 'box' in item:
                sum_w += item['box'][2]
                count_w += 1
        avg_w = (sum_w / count_w) if count_w > 0 else 100
        SCALE_FACTOR = (100.0 / avg_w) if count_w > 0 else 1.0

        # Map to keep track of the relationship between ai_data index and actual Component
        index_to_comp = {}

        # 2. Place all components snapped perfectly to the grid
        for idx, item in enumerate(ai_data):
            raw_type = item['type']
            if raw_type in NON_COMPONENT_TYPES: continue 

            sp_type = type_mapping.get(raw_type, raw_type)
            cx, cy = item['center']
            
            snapped_x = round((cx * SCALE_FACTOR) / GRID_SIZE) * GRID_SIZE
            snapped_y = round((cy * SCALE_FACTOR) / GRID_SIZE) * GRID_SIZE

            rotation = item.get('rotation', 0)
            if 'box' in item and not rotation:
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
            index_to_comp[idx] = comp

        def get_comp_by_idx(c_idx):
            if isinstance(c_idx, int):
                return index_to_comp.get(c_idx)
            s = str(c_idx)
            if s.startswith("comp_"):
                s = s[5:]
            if s.lstrip('-').isdigit():
                return index_to_comp.get(int(s))
            return index_to_comp.get(c_idx)

        def get_pin_coords(pin_ref):
            c_idx = pin_ref['comp_idx']
            p_id = pin_ref['pin_id']
            if str(c_idx) in ['-1', 'junc', 'junction'] or str(c_idx).startswith('junc'):
                # Junction
                if pin_anchors:
                    for anchor in pin_anchors:
                        if str(anchor.get('comp_idx')) in ['-1', 'junc', 'junction'] and str(anchor.get('pin_id')) == str(p_id):
                            jx = round((anchor['x'] * SCALE_FACTOR) / GRID_SIZE) * GRID_SIZE
                            jy = round((anchor['y'] * SCALE_FACTOR) / GRID_SIZE) * GRID_SIZE
                            return (jx, jy), None
                return None, None
            else:
                comp = get_comp_by_idx(c_idx)
                if comp:
                    pins = comp.get_pins()
                    pid_num = None
                    if isinstance(p_id, int):
                        pid_num = p_id
                    elif str(p_id).isdigit():
                        pid_num = int(p_id)
                    elif isinstance(p_id, str) and "_" in p_id and p_id.rsplit("_", 1)[-1].isdigit():
                        pid_num = int(p_id.rsplit("_", 1)[-1])

                    if pid_num is not None and 0 <= pid_num < len(pins):
                        return pins[pid_num], comp
            return None, None

        # 3. Wire Routing
        transparent_types = {'label', 'junction', 'crossover', 'terminal', 'gnd'}
        if connections is not None:
            # Use ML traced connections with A* grid pathfinder and multi-point detours
            for conn in connections:
                p1, comp1 = get_pin_coords(conn['pin1'])
                p2, comp2 = get_pin_coords(conn['pin2'])
                
                if p1 and p2:
                    if p1[0] == p2[0] and p1[1] == p2[1]:
                        continue
                    
                    exclude = [c for c in [comp1, comp2] if c is not None]
                    obstacle_comps = [c for c in self.components if c not in exclude and c.type not in transparent_types]

                    # Try A* pathfinder first
                    astar_segs = self.route_a_star(p1, p2, obstacle_comps)
                    if astar_segs:
                        self.wires.extend(astar_segs)
                    else:
                        detour_segs = self.route_around_component(p1, p2, obstacle_comps)
                        self.wires.extend(detour_segs)

            # Perform wire cleanup & collinear consolidation
            self.reroute_all_wires()
        else:
            # THE GRID-RAYCAST AUTO-ROUTER (Legacy Fallback)
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
                        dist = abs(y1 - y2) if is_vertical_align else abs(x1 - x2)
                        if dist > 500: continue
                        
                        blocked = False
                        for block_comp in self.components:
                            if block_comp in [c1, c2]: continue
                            if is_vertical_align:
                                if abs(block_comp.x - x1) < 25 and min(y1, y2) < block_comp.y < max(y1, y2):
                                    blocked = True; break
                            else:
                                if abs(block_comp.y - y1) < 25 and min(x1, x2) < block_comp.x < max(x1, x2):
                                    blocked = True; break
                                    
                        if not blocked:
                            self.wires.append(((x1, y1), (x2, y2)))

        # 4. Auto-Center Circuit on Canvas (Matching WebD canvas fitting)
        if self.components:
            min_x = min(c.x for c in self.components)
            max_x = max(c.x for c in self.components)
            min_y = min(c.y for c in self.components)
            max_y = max(c.y for c in self.components)

            target_cx, target_cy = 500, 350
            curr_cx, curr_cy = (min_x + max_x) / 2, (min_y + max_y) / 2

            dx = round((target_cx - curr_cx) / GRID_SIZE) * GRID_SIZE
            dy = round((target_cy - curr_cy) / GRID_SIZE) * GRID_SIZE

            for c in self.components:
                c.x += dx
                c.y += dy

            self.wires = [((w1[0] + dx, w1[1] + dy), (w2[0] + dx, w2[1] + dy)) for (w1, w2) in self.wires]

        self.auto_update_junctions()
        self.redraw_all()

    # ==========================================
    # SIMULATION ENGINE & NETLIST OVERRIDE
    # ==========================================

    def open_sim_dialog(self): 
        node_map, sources, unique_nodes, sweepables = analyze_circuit(self.components, self.wires)
        SimulationDialog(self.root, unique_nodes, sources, sweepables, self.sim_data, self.set_sim_data)

    def set_sim_data(self, data): 
        self.sim_data = data
        self.lbl_sim.config(text=data['cmd'])

    def open_netlist_dialog(self):
        """Open the Netlist Preview and Manual Override Editor Dialog."""
        win = tk.Toplevel(self.root)
        win.title("📄 SPICE Netlist Preview & Override Editor")
        win.geometry("750x550")
        win.configure(bg="#1E1E1E")

        top_frame = tk.Frame(win, bg="#1E1E1E", padx=15, pady=10)
        top_frame.pack(fill="x")

        tk.Label(top_frame, text="📄 SPICE Netlist Editor", bg="#1E1E1E", fg="#4CAF50", font=("Segoe UI", 12, "bold")).pack(side="left")

        override_var = tk.BooleanVar(value=getattr(self, 'manual_netlist_mode', False))

        def toggle_override():
            self.manual_netlist_mode = override_var.get()
            if self.manual_netlist_mode:
                txt.config(bg="#222222", fg="#00E5FF")
                self.status.config(text="⚠️ Manual Netlist Override Mode enabled.")
            else:
                txt.config(bg="#111111", fg="#E0E0E0")
                self.status.config(text="Ready (Auto-generated netlist mode)")

        chk = tk.Checkbutton(top_frame, text="Enable Manual SPICE Override", variable=override_var, command=toggle_override,
                             bg="#1E1E1E", fg="#FFD740", selectcolor="#2D2D2D", activebackground="#1E1E1E", activeforeground="#FFD740", font=("Segoe UI", 10, "bold"))
        chk.pack(side="right")

        text_frame = tk.Frame(win, bg="#1E1E1E", padx=15, pady=5)
        text_frame.pack(fill="both", expand=True)

        txt = tk.Text(text_frame, bg="#111111", fg="#E0E0E0", insertbackground="white", bd=0, highlightthickness=1, font=("Consolas", 10))
        txt.pack(fill="both", side="left", expand=True)

        scrollbar = tk.Scrollbar(text_frame, command=txt.yview)
        scrollbar.pack(side="right", fill="y")
        txt.config(yscrollcommand=scrollbar.set)

        current_netlist = getattr(self, 'manual_netlist_code', '') if getattr(self, 'manual_netlist_mode', False) else generate_netlist(self.components, self.wires, self.sim_data)
        txt.insert("1.0", current_netlist)

        if override_var.get():
            txt.config(bg="#222222", fg="#00E5FF")

        btn_frame = tk.Frame(win, bg="#1E1E1E", pady=10)
        btn_frame.pack(fill="x")

        def save_netlist():
            if override_var.get():
                self.manual_netlist_mode = True
                self.manual_netlist_code = txt.get("1.0", "end-1c")
                messagebox.showinfo("Netlist Saved", "Manual netlist override saved! Simulation will use this custom netlist.")
            else:
                self.manual_netlist_mode = False
                self.manual_netlist_code = ""
                messagebox.showinfo("Netlist Saved", "Reverted to auto-generated netlist mode.")
            win.destroy()

        tk.Button(btn_frame, text="Save & Apply Netlist", command=save_netlist, bg=COLOR_ACCENT_BLUE, fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=15, pady=5).pack(side="right", padx=15)
        tk.Button(btn_frame, text="Close", command=win.destroy, bg="#444444", fg="white", font=("Segoe UI", 10), relief="flat", padx=15, pady=5).pack(side="right")

    def check_ground_connected_labels(self):
        """Scans the circuit graph using DFS from Ground nodes to check if any named label component is directly connected to Ground."""
        gnd_comps = [c for c in self.components if c.type == 'gnd']
        if not gnd_comps:
            return True  # No ground components, skip check

        # Build adjacency map of grid points connected by wires
        adj = {}
        for (x1, y1), (x2, y2) in self.wires:
            adj.setdefault((x1, y1), set()).add((x2, y2))
            adj.setdefault((x2, y2), set()).add((x1, y1))

        # Add connections via zero-resistance component pins (junctions, etc.)
        for c in self.components:
            if c.type in ['junction', 'terminal', 'crossover']:
                for px, py in c.get_pins():
                    adj.setdefault((c.x, c.y), set()).add((px, py))
                    adj.setdefault((px, py), set()).add((c.x, c.y))

        # Run DFS from all ground node pin locations
        visited = set()
        stack = []
        for g in gnd_comps:
            for px, py in g.get_pins():
                stack.append((px, py))
                visited.add((px, py))

        while stack:
            curr = stack.pop()
            for nxt in adj.get(curr, []):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)

        # Check if any named label is at a visited ground coordinate
        shorted_labels = []
        for c in self.components:
            if c.type == 'label':
                if (c.x, c.y) in visited:
                    label_name = c.params.get('name', 'OUT')
                    shorted_labels.append(f"{c.name} ('{label_name}')")

        if shorted_labels:
            labels_str = ", ".join(shorted_labels)
            msg = f"⚠️ Pre-Simulation Warning:\nThe following node label(s) are directly shorted to Ground:\n\n  • {labels_str}\n\nDo you want to proceed with simulation anyway?"
            res = messagebox.askyesno("Ground Warning", msg, icon="warning")
            return res
        return True

    def run_simulation(self):
        if not self.check_ground_connected_labels():
            self.status.config(text="Simulation cancelled due to ground-shorted label warning.")
            return

        cwd = os.getcwd()
        filepath = os.path.join(cwd, "circuit.cir")

        if getattr(self, 'manual_netlist_mode', False) and getattr(self, 'manual_netlist_code', ''):
            netlist_code = self.manual_netlist_code
        else:
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

    def locate_target(self, x, y):
        """Center the viewport on target coordinates (x, y) and trigger pulsing highlight animation."""
        w = self.canvas.winfo_width() or 1000
        h = self.canvas.winfo_height() or 700
        self.offset_x = (w / 2) - (x * self.zoom)
        self.offset_y = (h / 2) - (y * self.zoom)
        self.redraw_all()
        self._pulse_canvas_highlight(x, y)

    def _pulse_canvas_highlight(self, x, y, step=0):
        """Draw an expanding/fading pulsing ring animation on the canvas centered at (x, y)."""
        self.canvas.delete("pulse_highlight")
        if step >= 15:
            return

        sx, sy = self.world_to_screen(x, y)
        r = (15 + step * 4) * self.zoom
        colors = ["#FF2222", "#FF5555", "#FF8888", "#FFAAAA", "#FFCCCC"]
        col = colors[min(step // 3, len(colors) - 1)]

        self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r,
                                outline=col, width=3, tags="pulse_highlight")
        self.root.after(60, lambda: self._pulse_canvas_highlight(x, y, step + 1))

    def show_diagnostics_dialog(self, logs):
        diag_win = tk.Toplevel(self.root)
        diag_win.title("Simulation Diagnostics Console")
        diag_win.geometry("800x500")
        diag_win.configure(bg="#1E1E1E")

        title_frame = tk.Frame(diag_win, bg="#1E1E1E")
        title_frame.pack(fill="x", pady=10, padx=15)
        tk.Label(title_frame, text="🔍 Simulation Diagnostics Console", bg="#1E1E1E", fg="#4CAF50", font=("Segoe UI", 12, "bold")).pack(side="left")

        # Scrollable list frame for diagnostic entries
        canvas = tk.Canvas(diag_win, bg="#111111", highlightthickness=0)
        scrollbar = tk.Scrollbar(diag_win, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg="#111111")

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="top", fill="both", expand=True, padx=15, pady=5)
        scrollbar.pack(side="right", fill="y")

        node_map, _, _, _ = analyze_circuit(self.components, self.wires)
        # Reverse lookup for node coordinates
        node_coords = {str(node): pt for pt, node in node_map.items()}

        for idx, log in enumerate(logs):
            entry_frame = tk.Frame(scroll_frame, bg="#1A1A1A", bd=1, relief="solid", padx=8, pady=6)
            entry_frame.pack(fill="x", expand=True, pady=4, padx=5)

            prefix = "[INFO] "
            color = "#888888"
            if log['type'] == 'error':
                prefix = "[ERROR] ❌ "
                color = "#FF5252"
            elif log['type'] == 'warning':
                prefix = "[WARNING] ⚠️ "
                color = "#FFD740"

            msg_text = f"{prefix}{log['message']}"
            if log['line']:
                msg_text += f" (Line {log['line']}: '{log['code'].strip()}')"

            hdr_lbl = tk.Label(entry_frame, text=msg_text, bg="#1A1A1A", fg=color, font=("Consolas", 9), anchor="w", justify="left", wraplength=520)
            hdr_lbl.pack(side="left", fill="x", expand=True)

            # Locate target coordinates
            target_pt = None
            target_desc = ""
            if log['component']:
                comp = next((c for c in self.components if c.name.lower() == log['component'].lower()), None)
                if comp:
                    target_pt = (comp.x, comp.y)
                    target_desc = f"Comp {comp.name}"
            elif log['node'] and str(log['node']) in node_coords:
                target_pt = node_coords[str(log['node'])]
                target_desc = f"Node {log['node']}"

            if target_pt:
                btn_loc = tk.Button(entry_frame, text=f"🔍 Locate {target_desc}",
                                    command=lambda pt=target_pt, w=diag_win: (w.destroy(), self.locate_target(pt[0], pt[1])),
                                    bg=COLOR_ACCENT_BLUE, fg="white", font=("Segoe UI", 8, "bold"), relief="flat", padx=8, pady=2)
                btn_loc.pack(side="right", padx=5)

        btn_close = tk.Button(diag_win, text="Dismiss Console", command=diag_win.destroy, bg="#444444", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=15, pady=5)
        btn_close.pack(pady=10)

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
            self.update_sidebar()
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
                  bg=COLOR_ACCENT_BLUE, fg="white", relief="flat", font=("Segoe UI", 9), padx=8, pady=3).pack(side="left", padx=2)
        tk.Button(g_tools, text="🔍 PROBE", command=lambda: self.set_mode("probe"), 
                  bg="#FF9800", fg="white", relief="flat", font=("Segoe UI", 9), padx=8, pady=3).pack(side="left", padx=2)
        tk.Button(g_tools, text="⏚ GND", command=lambda: self.set_mode('gnd'), 
                  bg="#43A047", fg="white", relief="flat", font=("Segoe UI", 9), padx=8, pady=3).pack(side="left", padx=2)
        tk.Button(g_tools, text="❖ JXN", command=lambda: self.set_mode('junction'),
                  bg="#F57F17", fg="white", relief="flat", font=("Segoe UI", 9), padx=8, pady=3).pack(side="left", padx=2)
        tk.Button(g_tools, text="↺ Reroute", command=self.reroute_all_wires,
                  bg="#6A1B9A", fg="white", relief="flat", font=("Segoe UI", 9), padx=8, pady=3).pack(side="left", padx=2)
        tk.Button(g_tools, text="↶ Undo", command=self.undo,
                  bg="#444444", fg="white", relief="flat", font=("Segoe UI", 9), padx=8, pady=3).pack(side="left", padx=2)
        tk.Button(g_tools, text="↷ Redo", command=self.redo,
                  bg="#444444", fg="white", relief="flat", font=("Segoe UI", 9), padx=8, pady=3).pack(side="left", padx=2)
        tk.Button(g_tools, text="🔍 Fit", command=self.fit_to_screen,
                  bg="#444444", fg="white", relief="flat", font=("Segoe UI", 9), padx=8, pady=3).pack(side="left", padx=2)

        # Spacer between drawing tools and simulation control
        tk.Frame(ribbon, width=15, bg=COLOR_TOOLBAR_BG).pack(side="left")

        # Simulation Control (Direct Buttons)
        g_sim = tk.Frame(ribbon, bg=COLOR_TOOLBAR_BG)
        g_sim.pack(side="left", pady=5)
        tk.Button(g_sim, text="Config", command=self.open_sim_dialog, 
                  bg="#555555", fg="white", relief="flat", font=("Segoe UI", 9), padx=10, pady=3).pack(side="left", padx=2)
        tk.Button(g_sim, text="📄 Netlist", command=self.open_netlist_dialog,
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
        # Junction tool: place a dot junction directly (bypasses generic create_component)
        if self.mode == 'junction': return self._handle_junction_click(wx, wy)
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
            self.save_state()
            sx, sy = self.wire_start
            if sx != wx:
                self.wires.append(((sx, sy), (wx, sy)))
                self._check_and_add_junction(sx, sy)
                self._check_and_add_junction(wx, sy)
            if sy != wy:
                self.wires.append(((wx, sy), (wx, wy)))
                self._check_and_add_junction(wx, sy)
                self._check_and_add_junction(wx, wy)
            self.auto_update_junctions()
            self.redraw_all()
            self.wire_start = None if self.hovered_pin else (wx, wy)
        else:
            self.wire_start = (wx, wy)

    def _handle_junction_click(self, wx, wy):
        """Place a junction dot at the clicked grid position.
        Prevents placing a duplicate on an existing junction at the same spot."""
        for c in self.components:
            if c.type == 'junction' and c.x == wx and c.y == wy:
                self.status.config(text="⚠️ Junction already exists here.")
                return
        self.create_junction(wx, wy)
        self.status.config(text=f"✦ Junction placed at ({wx}, {wy})")
        self.redraw_all()


    def _handle_selection_click(self, wx, wy):
        clicked_comp = next((c for c in self.components if abs(c.x-wx)<30 and abs(c.y-wy)<30), None)
        if clicked_comp:
            self.selected_comps, self.selected_wires = [clicked_comp], []
            self.drag_start_world = (wx, wy)
            self.drag_start_positions = {c: (c.x, c.y) for c in self.selected_comps}
            self.pre_existing_overlap_pairs = self._get_overlap_pairs()
            self.pre_existing_wire_overlaps = self._get_wire_overlaps()
            # Capture wires attached to pins of selected component(s) for live dragging
            self.drag_attached_wires = self._get_wires_attached_to_comp_pins(self.selected_comps)
            self.drag_start_wire_positions = {
                idx: (self.wires[idx][0], self.wires[idx][1])
                for idx in self.drag_attached_wires if idx < len(self.wires)
            }
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
                for c in self.selected_comps:
                    c.x += dx
                    c.y += dy
                # Move wire endpoints that are pinned to the dragged component(s)
                for w_idx, info in self.drag_attached_wires.items():
                    if w_idx < len(self.wires):
                        p1, p2 = self.wires[w_idx]
                        new_p1 = (p1[0] + dx, p1[1] + dy) if info['move_p1'] else p1
                        new_p2 = (p2[0] + dx, p2[1] + dy) if info['move_p2'] else p2
                        self.wires[w_idx] = (new_p1, new_p2)
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
                # Roll back component positions
                for c, pos in self.drag_start_positions.items():
                    c.x, c.y = pos
                # Roll back wire endpoint positions
                for idx, (p1, p2) in self.drag_start_wire_positions.items():
                    if idx < len(self.wires):
                        self.wires[idx] = (p1, p2)
                self.status.config(text="⚠️ Overlap detected! Rollback snapped.")
                self.redraw_all()
            else:
                self.status.config(text="Ready")
            
            self.drag_start_positions = {}
            self.pre_existing_overlap_pairs = set()
            self.pre_existing_wire_overlaps = set()
            self.drag_attached_wires = {}
            self.drag_start_wire_positions = {}

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
            
            if degree < 2:
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
            (x1, y1), (x2, y2) = wire
            if x1 == x2 or y1 == y2:
                p1, p2 = self.world_to_screen(x1, y1), self.world_to_screen(x2, y2)
                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=col, width=width)
            else:
                # Strictly orthogonal Manhattan rendering (Horizontal then Vertical)
                p1 = self.world_to_screen(x1, y1)
                pm = self.world_to_screen(x2, y1)
                p2 = self.world_to_screen(x2, y2)
                self.canvas.create_line(p1[0], p1[1], pm[0], pm[1], fill=col, width=width)
                self.canvas.create_line(pm[0], pm[1], p2[0], p2[1], fill=col, width=width)

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
    # STATE HISTORY (UNDO / REDO)
    # ==========================================

    def save_state(self):
        """Pushes current circuit state snapshot to undo stack and clears redo stack."""
        state = {
            'components': [
                {
                    'type': c.type, 'x': c.x, 'y': c.y, 'name': c.name,
                    'rotation': c.rotation, 'params': copy.deepcopy(c.params)
                } for c in self.components
            ],
            'wires': copy.deepcopy(self.wires),
            'counts': copy.deepcopy(self.counts)
        }
        self.undo_stack.append(state)
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        self.redo_stack = []

    def undo(self, event=None):
        if not self.undo_stack:
            self.status.config(text="⚠️ Nothing to undo.")
            return
        
        current_state = {
            'components': [
                {
                    'type': c.type, 'x': c.x, 'y': c.y, 'name': c.name,
                    'rotation': c.rotation, 'params': copy.deepcopy(c.params)
                } for c in self.components
            ],
            'wires': copy.deepcopy(self.wires),
            'counts': copy.deepcopy(self.counts)
        }
        self.redo_stack.append(current_state)

        prev_state = self.undo_stack.pop()
        self._restore_state(prev_state)
        self.status.config(text="↶ Undo performed.")
        self.redraw_all()

    def redo(self, event=None):
        if not self.redo_stack:
            self.status.config(text="⚠️ Nothing to redo.")
            return

        current_state = {
            'components': [
                {
                    'type': c.type, 'x': c.x, 'y': c.y, 'name': c.name,
                    'rotation': c.rotation, 'params': copy.deepcopy(c.params)
                } for c in self.components
            ],
            'wires': copy.deepcopy(self.wires),
            'counts': copy.deepcopy(self.counts)
        }
        self.undo_stack.append(current_state)

        next_state = self.redo_stack.pop()
        self._restore_state(next_state)
        self.status.config(text="↷ Redo performed.")
        self.redraw_all()

    def _restore_state(self, state):
        self.components = []
        for item in state['components']:
            c = Component(item['type'], item['x'], item['y'], item['name'])
            c.rotation = item['rotation']
            c.params = copy.deepcopy(item['params'])
            self.components.append(c)
        self.wires = copy.deepcopy(state['wires'])
        self.counts = copy.deepcopy(state['counts'])
        self.selected_comps = []
        self.selected_wires = []
        self.update_sidebar()

    def select_all(self, event=None):
        self.selected_comps = list(self.components)
        self.selected_wires = list(self.wires)
        self.update_sidebar()
        self.redraw_all()

    def duplicate_selection(self, event=None):
        if not self.selected_comps: return
        self.save_state()
        new_selection = []
        for c in self.selected_comps:
            prefix = DB[c.type]['prefix'] if c.type in DB else "U"
            self.counts[prefix] = self.counts.get(prefix, 0) + 1
            new_c = Component(c.type, c.x + 20, c.y + 20, f"{prefix}{self.counts[prefix]}")
            new_c.rotation = c.rotation
            new_c.params = copy.deepcopy(c.params)
            self.components.append(new_c)
            new_selection.append(new_c)
        self.selected_comps = new_selection
        self.selected_wires = []
        self.update_sidebar()
        self.redraw_all()

    def cancel_current_action(self, event=None):
        self.set_mode('select')
        self.wire_start = None
        self.canvas.delete("temp_wire", "ghost")
        self.status.config(text="Ready")

    def fit_to_screen(self, event=None):
        """Auto-fits and centers all components and wires inside the canvas window."""
        if not self.components: return
        min_x = min(c.x for c in self.components)
        max_x = max(c.x for c in self.components)
        min_y = min(c.y for c in self.components)
        max_y = max(c.y for c in self.components)
        for w in self.wires:
            min_x = min(min_x, w[0][0], w[1][0])
            max_x = max(max_x, w[0][0], w[1][0])
            min_y = min(min_y, w[0][1], w[1][1])
            max_y = max(max_y, w[0][1], w[1][1])

        cw = self.canvas.winfo_width() or 1000
        ch = self.canvas.winfo_height() or 700

        width = max(100, max_x - min_x)
        height = max(100, max_y - min_y)

        scale_x = (cw - 100) / width
        scale_y = (ch - 100) / height
        new_zoom = max(0.4, min(scale_x, scale_y, 2.0))

        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2

        self.zoom = new_zoom
        self.offset_x = (cw / 2) - (cx * self.zoom)
        self.offset_y = (ch / 2) - (cy * self.zoom)
        self.redraw_all()
        self.status.config(text="🔍 Zoom Fit Centered.")

    def export_schematic_image(self):
        """Export the current canvas view to a PNG image file."""
        filepath = filedialog.asksaveasfilename(title="Export Schematic Image", defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")])
        if not filepath: return
        try:
            from PIL import Image, ImageDraw
            min_x = min([c.x for c in self.components] + [w[0][0] for w in self.wires] + [100]) - 60
            max_x = max([c.x for c in self.components] + [w[0][0] for w in self.wires] + [900]) + 60
            min_y = min([c.y for c in self.components] + [w[0][1] for w in self.wires] + [100]) - 60
            max_y = max([c.y for c in self.components] + [w[0][1] for w in self.wires] + [600]) + 60

            w_img = int(max_x - min_x)
            h_img = int(max_y - min_y)
            img = Image.new("RGB", (w_img, h_img), "#1E1E1E")
            draw = ImageDraw.Draw(img)

            # Draw Wires
            for (x1, y1), (x2, y2) in self.wires:
                px1, py1 = int(x1 - min_x), int(y1 - min_y)
                px2, py2 = int(x2 - min_x), int(y2 - min_y)
                draw.line([(px1, py1), (px2, py2)], fill="#4FC1FF", width=2)

            # Draw Components
            for comp in self.components:
                _, pil_img = ComponentHelper.render_image(comp.type, comp.rotation, 1.0, comp)
                if pil_img:
                    cx, cy = int(comp.x - min_x), int(comp.y - min_y)
                    img.paste(pil_img, (cx - pil_img.width // 2, cy - pil_img.height // 2), pil_img)

            img.save(filepath)
            messagebox.showinfo("Export Complete", f"Schematic exported successfully to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export schematic image:\n{e}")

    def export_netlist_file(self):
        """Export the SPICE netlist to a user-selected .cir file."""
        filepath = filedialog.asksaveasfilename(title="Save SPICE Netlist", defaultextension=".cir", filetypes=[("SPICE Netlist", "*.cir"), ("Text File", "*.txt"), ("All Files", "*.*")])
        if not filepath: return
        try:
            netlist_code = generate_netlist(self.components, self.wires, self.sim_data)
            with open(filepath, "w") as f:
                f.write(netlist_code)
            messagebox.showinfo("Export Complete", f"SPICE netlist saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save netlist file:\n{e}")

    def show_shortcuts_dialog(self):
        win = tk.Toplevel(self.root)
        win.title("⌨️ PySpice Studio Keyboard Shortcuts")
        win.geometry("500x480")
        win.configure(bg="#1E1E1E")

        tk.Label(win, text="⌨️ Keyboard Shortcuts Reference", bg="#1E1E1E", fg="#4CAF50", font=("Segoe UI", 12, "bold")).pack(pady=15)

        txt = tk.Text(win, bg="#111111", fg="#E0E0E0", bd=0, highlightthickness=1, font=("Consolas", 10), padx=10, pady=10)
        txt.pack(fill="both", expand=True, padx=15, pady=5)

        shortcuts = [
            ("Tool Selection", ""),
            ("  W", "Select Wire Tool"),
            ("  P", "Select Probe Tool"),
            ("  G", "Place Ground (GND)"),
            ("  J", "Place Dot Junction"),
            ("  R / C / L / D", "Place Resistor / Cap / Inductor / Diode"),
            ("", ""),
            ("Editing & Canvas Controls", ""),
            ("  Ctrl + Z", "Undo action"),
            ("  Ctrl + Y / Ctrl+Shift+Z", "Redo action"),
            ("  Ctrl + A", "Select All"),
            ("  Ctrl + C / Ctrl + V", "Copy / Paste selection"),
            ("  Ctrl + D", "Duplicate selection"),
            ("  Ctrl + R / Right-Click", "Rotate selected component(s)"),
            ("  Delete", "Delete selection"),
            ("  Escape", "Cancel tool / Return to Select mode"),
            ("  F", "Fit circuit to screen"),
            ("  F1", "Show Shortcuts Window"),
        ]

        for key, desc in shortcuts:
            if not desc:
                txt.insert("end", f"\n{key}\n", "header")
            else:
                txt.insert("end", f"{key:<22} {desc}\n")

        txt.tag_config("header", foreground="#00E5FF", font=("Segoe UI", 10, "bold"))
        txt.config(state="disabled")

        tk.Button(win, text="Close", command=win.destroy, bg="#0078D7", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=15, pady=5).pack(pady=10)

    # ==========================================
    # STATE MODIFIERS
    # ==========================================

    def create_component(self, c_type, x, y, rotation):
        self.save_state()
        prefix = DB[c_type]['prefix'] if c_type in DB else "U"
        self.counts[prefix] = self.counts.get(prefix, 0) + 1
        comp = Component(c_type, x, y, f"{prefix}{self.counts[prefix]}")
        comp.rotation = rotation
        # Guard: block placement if it overlaps an existing component
        transparent_types = {'label', 'junction', 'crossover', 'terminal', 'gnd'}
        if c_type not in transparent_types:
            for existing in self.components:
                if existing.type in transparent_types:
                    continue
                if self.components_overlap(comp, existing):
                    self.status.config(text=f"⚠️ Cannot place: Overlaps {existing.name}")
                    return  # Abort placement
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
            self.save_state()
            transparent_types = {'label', 'junction', 'crossover', 'terminal', 'gnd'}
            for c in self.selected_comps:
                old_rotation = c.rotation
                c.rotation = (c.rotation + 90) % 360  # Tentatively rotate
                # Validate: check for collisions with the new bounding box
                collision = False
                if c.type not in transparent_types:
                    for other in self.components:
                        if other is c or other.type in transparent_types:
                            continue
                        if self.components_overlap(c, other):
                            collision = True
                            break
                    if not collision:
                        for w in self.wires:
                            if self.does_comp_overlap_wire(c, w):
                                collision = True
                                break
                if collision:
                    c.rotation = old_rotation  # Revert
                    self.status.config(text=f"⚠️ Cannot rotate {c.name}: Would overlap a component or wire.")
                else:
                    self.status.config(text="Ready")
            self.redraw_all()
        elif self.mode not in ['select', 'wire', 'probe']:
            self.ghost_rotation = (self.ghost_rotation + 90) % 360
            self.on_mouse_move(tk.Event())

    def delete_selection(self, e=None):
        if not self.selected_comps and not self.selected_wires: return
        self.save_state()
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
        self.save_state()
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
    # WIRE HELPERS
    # ==========================================

    def _get_wires_attached_to_comp_pins(self, comps):
        """Returns {wire_idx: {'move_p1': bool, 'move_p2': bool}} for all wires
        whose endpoints coincide with any pin of the given components."""
        pin_positions = set()
        for comp in comps:
            for px, py in comp.get_pins():
                pin_positions.add((px, py))

        attached = {}
        for idx, wire in enumerate(self.wires):
            p1, p2 = wire
            move_p1 = p1 in pin_positions
            move_p2 = p2 in pin_positions
            if move_p1 or move_p2:
                attached[idx] = {'move_p1': move_p1, 'move_p2': move_p2}
        return attached

    def _does_segment_intersect_component(self, p1, p2, obstacle_comps):
        (x1, y1), (x2, y2) = p1, p2
        transparent_types = {'label', 'junction', 'crossover', 'terminal', 'gnd'}
        for comp in obstacle_comps:
            if comp.type in transparent_types:
                continue
            w, h = self.get_hitbox(comp)
            bx1, by1 = comp.x - w / 2, comp.y - h / 2
            bx2, by2 = comp.x + w / 2, comp.y + h / 2
            
            if x1 == x2:  # Vertical segment
                if bx1 < x1 < bx2:
                    sy1, sy2 = min(y1, y2), max(y1, y2)
                    if sy1 < by2 and sy2 > by1:
                        return True
            elif y1 == y2:  # Horizontal segment
                if by1 < y1 < by2:
                    sx1, sx2 = min(x1, x2), max(x1, x2)
                    if sx1 < bx2 and sx2 > bx1:
                        return True
        return False

    def route_a_star(self, start, goal, obstacle_comps, grid_size=20):
        """Grid-based A* pathfinding algorithm between start and goal points with turn penalties."""
        sx, sy = round(start[0] / grid_size) * grid_size, round(start[1] / grid_size) * grid_size
        gx, gy = round(goal[0] / grid_size) * grid_size, round(goal[1] / grid_size) * grid_size

        if (sx, sy) == (gx, gy):
            return [((sx, sy), (gx, gy))]

        # Search area boundaries to prevent searching out to infinity
        margin = grid_size * 12
        min_x = min(sx, gx) - margin
        max_x = max(sx, gx) + margin
        min_y = min(sy, gy) - margin
        max_y = max(sy, gy) + margin

        transparent_types = {'label', 'junction', 'crossover', 'terminal', 'gnd'}
        obstacles = []
        for comp in obstacle_comps:
            if getattr(comp, 'type', '') in transparent_types:
                continue
            w, h = self.get_hitbox(comp)
            obstacles.append((comp.x - w / 2, comp.y - h / 2, comp.x + w / 2, comp.y + h / 2))

        def is_obstacle(x, y):
            if (x, y) == (sx, sy) or (x, y) == (gx, gy):
                return False
            for x1, y1, x2, y2 in obstacles:
                if x1 < x < x2 and y1 < y < y2:
                    return True
            return False

        def heuristic(x, y):
            return abs(x - gx) + abs(y - gy)

        turn_cost = grid_size * 3
        # State key: (x, y, dir_x, dir_y)
        start_state = (sx, sy, 0, 0)
        open_heap = [(heuristic(sx, sy), 0, start_state)]
        came_from = {}
        g_score = {start_state: 0}
        closed_set = set()

        found_state = None
        iterations = 0

        while open_heap and iterations < 1500:
            iterations += 1
            f, cost, curr_state = heapq.heappop(open_heap)
            cx, cy, cdx, cdy = curr_state

            if curr_state in closed_set:
                continue
            closed_set.add(curr_state)

            if (cx, cy) == (gx, gy):
                found_state = curr_state
                break

            for dx, dy in [(-grid_size, 0), (grid_size, 0), (0, -grid_size), (0, grid_size)]:
                nx, ny = cx + dx, cy + dy

                # Check bounding limits
                if not (min_x <= nx <= max_x and min_y <= ny <= max_y):
                    continue

                if is_obstacle(nx, ny):
                    continue

                move_cost = grid_size
                if (cdx, cdy) != (0, 0) and (dx, dy) != (cdx, cdy):
                    move_cost += turn_cost

                next_state = (nx, ny, dx, dy)
                if next_state in closed_set:
                    continue

                new_g = cost + move_cost
                if new_g < g_score.get(next_state, float('inf')):
                    g_score[next_state] = new_g
                    came_from[next_state] = curr_state
                    f_score = new_g + heuristic(nx, ny)
                    heapq.heappush(open_heap, (f_score, new_g, next_state))

        if not found_state:
            return None

        # Reconstruct path safely
        path_nodes = []
        curr = found_state
        visited_states = set()
        while curr in came_from:
            if curr in visited_states:
                break  # Prevent infinite loop
            visited_states.add(curr)
            path_nodes.append((curr[0], curr[1]))
            curr = came_from[curr]
        path_nodes.append((sx, sy))
        path_nodes.reverse()

        # Remove consecutive duplicate points
        dedup_nodes = []
        for pt in path_nodes:
            if not dedup_nodes or dedup_nodes[-1] != pt:
                dedup_nodes.append(pt)
        path_nodes = dedup_nodes

        if len(path_nodes) < 2:
            return []

        # Simplify collinear segments
        segments = []
        seg_start = path_nodes[0]
        for i in range(1, len(path_nodes) - 1):
            prev_p = path_nodes[i - 1]
            curr_p = path_nodes[i]
            next_p = path_nodes[i + 1]
            dir1 = (curr_p[0] - prev_p[0], curr_p[1] - prev_p[1])
            dir2 = (next_p[0] - curr_p[0], next_p[1] - curr_p[1])
            if dir1 != dir2:
                segments.append((seg_start, curr_p))
                seg_start = curr_p
        segments.append((seg_start, path_nodes[-1]))
        return segments

    def route_around_component(self, p1, p2, obstacle_comps):
        """Generates multi-point detours around component hitboxes when straight lines collide."""
        (x1, y1), (x2, y2) = p1, p2
        transparent_types = {'label', 'junction', 'crossover', 'terminal', 'gnd'}
        for comp in obstacle_comps:
            if comp.type in transparent_types:
                continue
            w, h = self.get_hitbox(comp)
            bx1, by1 = comp.x - w / 2 - 20, comp.y - h / 2 - 20
            bx2, by2 = comp.x + w / 2 + 20, comp.y + h / 2 + 20

            if x1 == x2 and bx1 <= x1 <= bx2 and min(y1, y2) < by2 and max(y1, y2) > by1:
                detour_x = bx1 if abs(x1 - bx1) < abs(x1 - bx2) else bx2
                return [
                    ((x1, y1), (x1, by1 if y1 < y2 else by2)),
                    ((x1, by1 if y1 < y2 else by2), (detour_x, by1 if y1 < y2 else by2)),
                    ((detour_x, by1 if y1 < y2 else by2), (detour_x, by2 if y1 < y2 else by1)),
                    ((detour_x, by2 if y1 < y2 else by1), (x2, by2 if y1 < y2 else by1)),
                    ((x2, by2 if y1 < y2 else by1), (x2, y2))
                ]
            elif y1 == y2 and by1 <= y1 <= by2 and min(x1, x2) < bx2 and max(x1, x2) > bx1:
                detour_y = by1 if abs(y1 - by1) < abs(y1 - by2) else by2
                return [
                    ((x1, y1), (bx1 if x1 < x2 else bx2, y1)),
                    ((bx1 if x1 < x2 else bx2, y1), (bx1 if x1 < x2 else bx2, detour_y)),
                    ((bx1 if x1 < x2 else bx2, detour_y), (bx2 if x1 < x2 else bx1, detour_y)),
                    ((bx2 if x1 < x2 else bx1, detour_y), (bx2 if x1 < x2 else bx1, y2)),
                    ((bx2 if x1 < x2 else bx1, y2), (x2, y2))
                ]

        if x1 != x2 and y1 != y2:
            mid = (x2, y1)
            return [((x1, y1), mid), (mid, (x2, y2))]
        return [(p1, p2)]

    def auto_update_junctions(self):
        """Automatically converts 2 or more meeting wire segments (corners, T-intersections, 3+ wire endpoints)
        into connected junction dots on the canvas, while enforcing strictly horizontal or vertical wires."""
        if not self.wires:
            return

        from collections import defaultdict

        # Step 0: Convert any non-orthogonal wire segment into strictly horizontal & vertical segments
        ortho_wires = []
        for (x1, y1), (x2, y2) in self.wires:
            if x1 == x2 or y1 == y2:
                ortho_wires.append(((x1, y1), (x2, y2)))
            else:
                mid = (x2, y1)
                ortho_wires.append(((x1, y1), mid))
                ortho_wires.append((mid, (x2, y2)))
        self.wires = [w for w in ortho_wires if w[0] != w[1]]

        # 1. Split wires where an endpoint touches the interior of another wire segment
        all_pts = set()
        for p1, p2 in self.wires:
            all_pts.add(p1)
            all_pts.add(p2)

        split_wires = []
        for p1, p2 in self.wires:
            x1, y1 = p1
            x2, y2 = p2
            inter_pts = []
            for px, py in all_pts:
                if (px, py) == p1 or (px, py) == p2:
                    continue
                if x1 == x2 == px and min(y1, y2) < py < max(y1, y2):
                    inter_pts.append((px, py))
                elif y1 == y2 == py and min(x1, x2) < px < max(x1, x2):
                    inter_pts.append((px, py))

            if inter_pts:
                if x1 == x2:
                    inter_pts.sort(key=lambda pt: pt[1], reverse=(y2 < y1))
                else:
                    inter_pts.sort(key=lambda pt: pt[0], reverse=(x2 < x1))

                curr = p1
                for ipt in inter_pts:
                    if curr != ipt:
                        split_wires.append((curr, ipt))
                    curr = ipt
                if curr != p2:
                    split_wires.append((curr, p2))
            else:
                split_wires.append((p1, p2))

        self.wires = [w for w in split_wires if w[0] != w[1]]

        # 2. Count wire segment connections at each endpoint
        endpoint_counts = defaultdict(int)
        for p1, p2 in self.wires:
            endpoint_counts[p1] += 1
            endpoint_counts[p2] += 1

        # Collect non-junction component pin coordinates
        comp_pin_coords = set()
        for comp in self.components:
            if comp.type != 'junction':
                for px, py in comp.get_pins():
                    comp_pin_coords.add((px, py))

        # 3. Place junction components at points with 2+ wire connections (including corners)
        existing_junctions = {(c.x, c.y): c for c in self.components if c.type == 'junction'}
        needed_junctions = {pt for pt, count in endpoint_counts.items() if count >= 2 and pt not in comp_pin_coords}

        # Remove junctions that no longer have 2+ wire connections
        for pos, junc_comp in list(existing_junctions.items()):
            if pos not in needed_junctions and endpoint_counts[pos] < 2:
                if junc_comp in self.components:
                    self.components.remove(junc_comp)

        for pt in needed_junctions:
            if pt not in existing_junctions:
                self.create_junction(pt[0], pt[1])

    def reroute_all_wires(self):
        """Clean up wires: remove zero-length/duplicate segments, merge collinear
        touching segments, then re-route any segment going through a component body using A* pathfinding."""
        if not self.wires:
            self.status.config(text="⚠️ No wires to reroute.")
            return

        # Step 1: Remove zero-length wires
        wires = [w for w in self.wires if w[0] != w[1]]

        # Step 2: Remove exact duplicates (normalize endpoint order)
        seen = set()
        deduped = []
        for w in wires:
            p1, p2 = w
            key = (p1, p2) if (p1[0], p1[1]) <= (p2[0], p2[1]) else (p2, p1)
            if key not in seen:
                seen.add(key)
                deduped.append(w)
        wires = deduped

        # Step 3: Merge collinear touching segments (multiple passes until stable)
        for _ in range(30):
            used = set()
            result = []
            merged_anything = False
            for i in range(len(wires)):
                if i in used:
                    continue
                p1, p2 = wires[i]
                for j in range(i + 1, len(wires)):
                    if j in used:
                        continue
                    q1, q2 = wires[j]
                    # Both horizontal on the same Y
                    if p1[1] == p2[1] == q1[1] == q2[1]:
                        if {p1[0], p2[0]} & {q1[0], q2[0]}:  # share an X endpoint
                            y = p1[1]
                            xs = sorted({p1[0], p2[0], q1[0], q2[0]})
                            p1, p2 = (xs[0], y), (xs[-1], y)
                            used.add(j)
                            merged_anything = True
                    # Both vertical on the same X
                    elif p1[0] == p2[0] == q1[0] == q2[0]:
                        if {p1[1], p2[1]} & {q1[1], q2[1]}:  # share a Y endpoint
                            x = p1[0]
                            ys = sorted({p1[1], p2[1], q1[1], q2[1]})
                            p1, p2 = (x, ys[0]), (x, ys[-1])
                            used.add(j)
                            merged_anything = True
                result.append((p1, p2))
            wires = result
            if not merged_anything:
                break

        # Step 4: Re-route segments that pass through component hitboxes using A* or Detours
        transparent_types = {'label', 'junction', 'crossover', 'terminal', 'gnd'}
        obstacle_comps = [c for c in self.components if c.type not in transparent_types]

        rerouted = []
        for p1, p2 in wires:
            if self._does_segment_intersect_component(p1, p2, obstacle_comps):
                # Try A* pathfinding router first
                astar_segs = self.route_a_star(p1, p2, obstacle_comps)
                if astar_segs:
                    rerouted.extend(astar_segs)
                else:
                    detour_segs = self.route_around_component(p1, p2, obstacle_comps)
                    rerouted.extend(detour_segs)
            else:
                rerouted.append((p1, p2))

        self.wires = rerouted
        self.auto_update_junctions()
        self.status.config(text=f"✅ Rerouted: {len(self.wires)} wire segment(s).")
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
        if self.mode == 'junction':
            # Draw a crisp dot ghost for junction placement
            r = max(5, int(7 * self.zoom))
            self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r,
                                    fill="#FFA726", outline="white", width=2,
                                    tags="ghost")
            return
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