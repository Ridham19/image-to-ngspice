import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import os

# --- Configuration ---
YOLO_MODEL_PATH = "epoch_40.pt"

class YoloTesterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Model Tester with Zoom & Pan")
        self.root.geometry("1100x850")
        self.root.configure(bg="#2c3e50")

        self.processed_pil_img = None
        self.zoom_scale = 1.0

        # Load the YOLO model
        self.status_var = tk.StringVar(value=f"⏳ Loading YOLO Model ({YOLO_MODEL_PATH})...")
        self.status = tk.Label(root, textvariable=self.status_var, font=("Arial", 11), fg="#ecf0f1", bg="#34495e", anchor="w", padx=10)
        self.status.pack(side="bottom", fill="x")
        self.root.update()

        try:
            self.model = YOLO(YOLO_MODEL_PATH)
            self.status_var.set(f"✅ Loaded {YOLO_MODEL_PATH} successfully. Ready to test.")
        except Exception as e:
            self.status_var.set(f"❌ Error loading model: {str(e)}")

        # UI Title Elements
        self.label_title = tk.Label(root, text="YOLO Model Prediction Viewer", font=("Arial", 20, "bold"), fg="white", bg="#2c3e50")
        self.label_title.pack(pady=15)

        # Controls Frame
        self.control_frame = tk.Frame(root, bg="#2c3e50")
        self.control_frame.pack(pady=5)

        self.btn_browse = tk.Button(self.control_frame, text="📂 Upload & Predict Image", command=self.browse_and_predict, 
                                  font=("Arial", 12, "bold"), bg="#3498db", fg="white", padx=15, pady=6, activebackground="#2980b9", activeforeground="white")
        self.btn_browse.grid(row=0, column=0, padx=5)

        self.btn_zoom_in = tk.Button(self.control_frame, text="🔍 Zoom In (+)", command=self.zoom_in, 
                                   font=("Arial", 11, "bold"), bg="#2ecc71", fg="white", padx=10, pady=5)
        self.btn_zoom_in.grid(row=0, column=1, padx=5)

        self.btn_zoom_out = tk.Button(self.control_frame, text="🔍 Zoom Out (-)", command=self.zoom_out, 
                                    font=("Arial", 11, "bold"), bg="#e74c3c", fg="white", padx=10, pady=5)
        self.btn_zoom_out.grid(row=0, column=2, padx=5)

        self.btn_reset = tk.Button(self.control_frame, text="🔄 Reset Zoom", command=self.reset_zoom, 
                                 font=("Arial", 11, "bold"), bg="#95a5a6", fg="white", padx=10, pady=5)
        self.btn_reset.grid(row=0, column=3, padx=5)

        # Instruction label for zoom/pan
        self.lbl_info = tk.Label(root, text="🖱️ Drag with Left Mouse Button to Pan | Use Mouse Wheel or Buttons to Zoom", font=("Arial", 10, "italic"), fg="#bdc3c7", bg="#2c3e50")
        self.lbl_info.pack(pady=2)

        # Canvas for displaying image with Scrollbars
        self.canvas_frame = tk.Frame(root, bg="#34495e", bd=2, relief="sunken")
        self.canvas_frame.pack(pady=10, expand=True, fill="both", padx=20)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#2f3542", highlightthickness=0)
        self.canvas.pack(side="left", expand=True, fill="both")

        self.hbar = tk.Scrollbar(self.canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.hbar.pack(side="bottom", fill="x")
        self.vbar = tk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.vbar.pack(side="right", fill="y")

        self.canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        # Canvas Event Bindings for Panning
        self.canvas.bind("<ButtonPress-1>", self.on_pan_start)
        self.canvas.bind("<B1-Motion>", self.on_pan_drag)
        # Mouse wheel zoom
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)

    def browse_and_predict(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.webp *.JPG")])
        if not file_path:
            return

        self.status_var.set(f"Running inference on {os.path.basename(file_path)}...")
        self.root.update_idletasks()

        # Read original image
        img = cv2.imread(file_path)
        if img is None:
            self.status_var.set("❌ Error: Could not read selected image.")
            return

        # Run detection using epoch_40.pt
        try:
            results = self.model.predict(img, conf=0.25, verbose=False)
            output_img = img.copy()
            detected_count = 0

            # Draw detections
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cls_id = int(box.cls[0])
                        label = self.model.names[cls_id]
                        conf = float(box.conf[0])
                        detected_count += 1

                        # Draw bold bounding box
                        cv2.rectangle(output_img, (x1, y1), (x2, y2), (46, 204, 113), 3)

                        # Draw high visibility filled background for text label
                        display_text = f"{label} ({conf:.2f})"
                        font_scale = 0.85
                        font_thickness = 2
                        text_size, _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                        text_w, text_h = text_size
                        
                        # Filled background box
                        cv2.rectangle(output_img, (x1, y1 - text_h - 14), (x1 + text_w + 10, y1), (46, 204, 113), -1)
                        # Text drawn cleanly on top
                        cv2.putText(output_img, display_text, (x1 + 5, y1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)

            # Convert BGR to RGB for PIL / Tkinter
            res_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            self.processed_pil_img = Image.fromarray(res_rgb)
            
            # Reset zoom and show
            self.zoom_scale = 1.0
            self.show_image()
            
            self.status_var.set(f"✅ Prediction complete! Detected {detected_count} objects.")

        except Exception as e:
            self.status_var.set(f"❌ Error during prediction: {str(e)}")

    def show_image(self):
        if self.processed_pil_img is None:
            return

        w, h = self.processed_pil_img.size
        new_w = int(w * self.zoom_scale)
        new_h = int(h * self.zoom_scale)

        resized_img = self.processed_pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized_img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.canvas.config(scrollregion=(0, 0, new_w, new_h))

    # Zoom functions
    def zoom_in(self):
        if self.processed_pil_img:
            self.zoom_scale *= 1.2
            self.show_image()

    def zoom_out(self):
        if self.processed_pil_img:
            self.zoom_scale /= 1.2
            if self.zoom_scale < 0.1:
                self.zoom_scale = 0.1
            self.show_image()

    def reset_zoom(self):
        if self.processed_pil_img:
            self.zoom_scale = 1.0
            self.show_image()

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    # Pan functions
    def on_pan_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def on_pan_drag(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloTesterGUI(root)
    root.mainloop()


