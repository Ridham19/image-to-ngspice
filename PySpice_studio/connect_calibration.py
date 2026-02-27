import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
from components import ComponentHelper
from library import DB

# --- CONFIG ---
CANVAS_SIZE = 600
BG_COLOR = "#1E1E1E"
GRID_COLOR = "#333333"

class Calibrator:
    def __init__(self, root):
        self.root = root
        self.root.title("Component Pin Calibrator")
        self.root.geometry("900x700")
        
        # State
        self.current_comp = None
        self.zoom = 5.0 # High zoom for precision
        self.img_ref = None
        
        # Layout
        self.setup_ui()
        self.populate_list()
        
    def setup_ui(self):
        # Left Panel (List)
        left = tk.Frame(self.root, width=200, bg="#252526")
        left.pack(side="left", fill="y")
        
        tk.Label(left, text="Components", fg="white", bg="#252526", font=("bold", 12)).pack(pady=10)
        self.listbox = tk.Listbox(left, bg="#333", fg="white", selectbackground="#0078D7", bd=0)
        self.listbox.pack(fill="both", expand=True, padx=5, pady=5)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        
        # Right Panel (Canvas + Info)
        right = tk.Frame(self.root, bg=BG_COLOR)
        right.pack(side="right", fill="both", expand=True)
        
        # Info Header
        self.lbl_info = tk.Label(right, text="Select a component...", bg=BG_COLOR, fg="yellow", font=("Consolas", 14))
        self.lbl_info.pack(pady=10)
        
        # Canvas
        self.canvas_frame = tk.Frame(right, width=CANVAS_SIZE, height=CANVAS_SIZE)
        self.canvas_frame.pack()
        
        self.canvas = tk.Canvas(self.canvas_frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bg=BG_COLOR, highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        
        # Zoom Control
        ctrl = tk.Frame(right, bg=BG_COLOR)
        ctrl.pack(pady=10)
        tk.Label(ctrl, text="Zoom Level:", fg="white", bg=BG_COLOR).pack(side="left")
        self.scl_zoom = tk.Scale(ctrl, from_=1, to=10, orient="horizontal", bg=BG_COLOR, fg="white", command=self.update_view)
        self.scl_zoom.set(5)
        self.scl_zoom.pack(side="left", padx=10)

        # Instructions
        tk.Label(right, text="Click on the image to get Pin Coordinates.\nRed Cross = Center (0,0)", fg="#888", bg=BG_COLOR).pack(side="bottom", pady=10)

    def populate_list(self):
        for key in sorted(DB.keys()):
            self.listbox.insert(tk.END, key)

    def on_select(self, event):
        sel = self.listbox.curselection()
        if sel:
            self.current_comp = self.listbox.get(sel[0])
            self.update_view()

    def update_view(self, val=None):
        if not self.current_comp: return
        
        self.zoom = self.scl_zoom.get()
        self.canvas.delete("all")
        
        # Draw Center Lines
        c = CANVAS_SIZE // 2
        self.canvas.create_line(c, 0, c, CANVAS_SIZE, fill=GRID_COLOR)
        self.canvas.create_line(0, c, CANVAS_SIZE, c, fill=GRID_COLOR)
        self.canvas.create_line(c-10, c, c+10, c, fill="red", width=2) # Center Marker
        self.canvas.create_line(c, c-10, c, c+10, fill="red", width=2)

        # Render Component
        # We assume the component logic renders it centered on a square tile
        # The tile size is (100 * zoom) in components.py
        # We want to display that tile centered on our canvas
        
        try:
            # Call the actual render logic
            tk_img, pil_img = ComponentHelper.render_image(self.current_comp, rotation=0, zoom_scale=self.zoom)
            self.img_ref = tk_img # Keep reference
            
            # Place image exactly in center of canvas
            self.canvas.create_image(c, c, image=tk_img, anchor="center")
            
        except Exception as e:
            print(e)

    def on_click(self, event):
        if not self.current_comp: return
        
        # Canvas Center
        c_x = CANVAS_SIZE // 2
        c_y = CANVAS_SIZE // 2
        
        # Click coordinates relative to Canvas Center
        rel_x = event.x - c_x
        rel_y = event.y - c_y
        
        # Convert back to Base Scale (Zoom = 1.0)
        # Because we rendered at self.zoom, the pixels are multiplied.
        # We divide by zoom to get the "Code Value".
        
        base_x = rel_x / self.zoom
        base_y = rel_y / self.zoom
        
        # Snap to nearest 5 or 10 for clean numbers
        snap_x = round(base_x / 5) * 5
        snap_y = round(base_y / 5) * 5
        
        # Display
        res = f"offset: ({int(snap_x)}, {int(snap_y)})"
        self.lbl_info.config(text=f"{self.current_comp} -> {res}")
        
        # Visual Marker
        r = 5
        self.canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r, fill="#00FF00", outline="black")
        self.canvas.create_text(event.x+10, event.y, text=f"({int(snap_x)}, {int(snap_y)})", anchor="w", fill="#00FF00", font=("Arial", 10, "bold"))
        
        print(f"['{self.current_comp}'] Pin: snap({int(snap_x)}, {int(snap_y)})")

if __name__ == "__main__":
    root = tk.Tk()
    app = Calibrator(root)
    root.mainloop()