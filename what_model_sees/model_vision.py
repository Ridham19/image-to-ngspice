import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics is not installed. Run 'pip install ultralytics'")
    exit()

class VisionTesterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Vision Tester - Bounding Boxes & Confidence Heatmap")
        self.root.configure(bg="#1E1E1E")
        self.root.geometry("1200x700")

        # --- LOAD MODEL ONCE ---
        self.model_path = "best.pt"
        self.model = None
        self._load_model()

        # --- UI SETUP ---
        # Top Toolbar
        self.toolbar = tk.Frame(self.root, bg="#2D2D2D", pady=10)
        self.toolbar.pack(side="top", fill="x")

        self.btn_load = tk.Button(self.toolbar, text="📂 Load New Image", command=self.process_image,
                                  bg="#0078D7", fg="white", font=("Segoe UI", 12, "bold"), padx=20)
        self.btn_load.pack(side="left", padx=20)

        self.lbl_status = tk.Label(self.toolbar, text="Ready. Please load an image.", 
                                   bg="#2D2D2D", fg="#888", font=("Segoe UI", 11))
        self.lbl_status.pack(side="left", padx=20)

        # Main Image Display Area
        self.canvas_frame = tk.Frame(self.root, bg="#1E1E1E")
        self.canvas_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.lbl_image = tk.Label(self.canvas_frame, bg="#1E1E1E")
        self.lbl_image.pack(expand=True)

    def _load_model(self):
        if not os.path.exists(self.model_path):
            messagebox.showerror("Model Not Found", f"Cannot find '{self.model_path}' in the current directory!")
            self.root.quit()
            return
            
        self.lbl_status = tk.Label(self.root, text="Loading AI Model... Please wait.", fg="white", bg="#1E1E1E")
        self.lbl_status.pack(pady=20)
        self.root.update()
        
        print("🧠 Loading AI Model into memory...")
        self.model = YOLO(self.model_path)
        self.lbl_status.destroy()

    def process_image(self):
        if not self.model: return

        # 1. Ask the user to select a test image
        image_path = filedialog.askopenfilename(
            title="Select Test Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")]
        )

        if not image_path:
            return

        self.lbl_status.config(text=f"👁️ Analyzing: {os.path.basename(image_path)}...")
        self.root.update()

        try:
            # 2. Run Inference
            results = self.model.predict(source=image_path, conf=0.40, show=False)

            # Extract base images
            original_img = results[0].orig_img
            annotated_img = results[0].plot()

            # ---------------------------------------------------------
            # 3. CREATE THE CONFIDENCE HEATMAP
            # ---------------------------------------------------------
            h_img, w_img = original_img.shape[:2]
            heat_mask = np.zeros((h_img, w_img), dtype=np.uint8)

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                conf = float(box.conf[0])
                
                # Size of the heat aura
                radius = max(30, int(max(x2-x1, y2-y1) * 0.6))
                
                # Intensity based on YOLO's confidence
                intensity = int(255 * conf)
                
                temp_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                cv2.circle(temp_mask, (cx, cy), radius, intensity, -1)
                heat_mask = cv2.add(heat_mask, temp_mask)

            # Blur to create smooth gradient
            kernel_size = min(w_img, h_img) // 8
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1 
            
            if kernel_size > 0:
                heat_mask = cv2.GaussianBlur(heat_mask, (kernel_size, kernel_size), 0)

            # Apply thermal palette and overlay
            heatmap_color = cv2.applyColorMap(heat_mask, cv2.COLORMAP_JET)
            heatmap_overlay = cv2.addWeighted(original_img, 0.4, heatmap_color, 0.6, 0)
            # ---------------------------------------------------------

            # 4. Stitch Annotated Image and Heatmap side-by-side
            combined_img = cv2.hconcat([annotated_img, heatmap_overlay])

            # 5. Convert OpenCV BGR to RGB for PIL
            rgb_img = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)

            # 6. Smart resize so it fits neatly inside the Tkinter window
            # Get the available screen/window size dynamically
            max_w = self.root.winfo_width() - 40
            max_h = self.root.winfo_height() - 100
            
            # Fallbacks just in case window isn't fully rendered
            if max_w < 100: max_w = 1400
            if max_h < 100: max_h = 700

            im_w, im_h = pil_img.size
            if im_h > max_h or im_w > max_w:
                scale = min(max_w/im_w, max_h/im_h)
                pil_img = pil_img.resize((int(im_w*scale), int(im_h*scale)), Image.Resampling.LANCZOS)

            # 7. Update the UI
            tk_img = ImageTk.PhotoImage(pil_img)
            self.lbl_image.config(image=tk_img)
            self.lbl_image.image = tk_img # VERY IMPORTANT: Keep reference to prevent garbage collection!
            
            self.lbl_status.config(text=f"✅ Done! Found {len(results[0].boxes)} objects.")

        except Exception as e:
            self.lbl_status.config(text="❌ Error during processing.")
            messagebox.showerror("Processing Error", str(e))

def main():
    root = tk.Tk()
    
    # Try to maximize window on launch
    try:
        root.state('zoomed')
    except:
        pass
        
    app = VisionTesterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()