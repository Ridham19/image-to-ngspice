import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# --- NEW IMPORTS FOR MATPLOTLIB IN TKINTER ---
import matplotlib
matplotlib.use("TkAgg")  # Tell Matplotlib to play nicely with Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics is not installed. Run 'pip install ultralytics'")
    exit()

class VisionTesterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Vision Tester - Main Engine")
        self.root.configure(bg="#1E1E1E")
        self.root.geometry("1200x700")

        # Keep track of extra windows so we can close old ones
        self.extra_windows = []

        # --- LOAD MODEL ONCE ---
        self.model_path = "best.pt"
        self.model = None
        self._load_model()

        # --- UI SETUP ---
        self.toolbar = tk.Frame(self.root, bg="#2D2D2D", pady=10)
        self.toolbar.pack(side="top", fill="x")

        self.btn_load = tk.Button(self.toolbar, text="📂 Load New Image", command=self.process_image,
                                  bg="#0078D7", fg="white", font=("Segoe UI", 12, "bold"), padx=20)
        self.btn_load.pack(side="left", padx=20)

        self.lbl_status = tk.Label(self.toolbar, text="Ready. Please load an image.", 
                                   bg="#2D2D2D", fg="#888", font=("Segoe UI", 11))
        self.lbl_status.pack(side="left", padx=20)

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

        image_path = filedialog.askopenfilename(
            title="Select Test Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")]
        )

        if not image_path:
            return

        # ---------------------------------------------------------
        # NEW: FOLDER CREATION LOGIC
        # ---------------------------------------------------------
        # Extract the base name of the image without the extension (e.g., "my_sketch" from "my_sketch.png")
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Create the specific directory path
        save_dir = os.path.join("vision_output", base_filename)
        
        # Make the directories if they don't exist
        os.makedirs(save_dir, exist_ok=True)
        # ---------------------------------------------------------

        self.lbl_status.config(text=f"👁️ Analyzing: {os.path.basename(image_path)}...")
        self.root.update()

        # Close old popup windows if the user loads a 2nd image
        for win in self.extra_windows:
            win.destroy()
        self.extra_windows.clear()

        try:
            # ---------------------------------------------------------
            # 1. THE YOLO AI INFERENCE
            # ---------------------------------------------------------
            results = self.model.predict(source=image_path, conf=0.40, show=False)

            original_img = results[0].orig_img
            annotated_img = results[0].plot()

            # Confidence Heatmap
            h_img, w_img = original_img.shape[:2]
            heat_mask = np.zeros((h_img, w_img), dtype=np.uint8)

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                conf = float(box.conf[0])
                radius = max(30, int(max(x2-x1, y2-y1) * 0.6))
                intensity = int(255 * conf)
                temp_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                cv2.circle(temp_mask, (cx, cy), radius, intensity, -1)
                heat_mask = cv2.add(heat_mask, temp_mask)

            kernel_size = min(w_img, h_img) // 8
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1 
            if kernel_size > 0: heat_mask = cv2.GaussianBlur(heat_mask, (kernel_size, kernel_size), 0)

            heatmap_color = cv2.applyColorMap(heat_mask, cv2.COLORMAP_JET)
            heatmap_overlay = cv2.addWeighted(original_img, 0.4, heatmap_color, 0.6, 0)
            combined_img = cv2.hconcat([annotated_img, heatmap_overlay])

            rgb_img = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)

            # --- SAVE THE MAIN YOLO OUTPUT ---
            yolo_save_path = os.path.join(save_dir, "0_yolo_heatmap.png")
            pil_img.save(yolo_save_path)

            max_w = self.root.winfo_width() - 40
            max_h = self.root.winfo_height() - 100
            if max_w < 100: max_w = 1400
            if max_h < 100: max_h = 700

            im_w, im_h = pil_img.size
            if im_h > max_h or im_w > max_w:
                scale = min(max_w/im_w, max_h/im_h)
                pil_img = pil_img.resize((int(im_w*scale), int(im_h*scale)), Image.Resampling.LANCZOS)

            tk_img = ImageTk.PhotoImage(pil_img)
            self.lbl_image.config(image=tk_img)
            self.lbl_image.image = tk_img 
            
            # ---------------------------------------------------------
            # 2. GENERATE AND OPEN THE EXTRA REPORT WINDOWS
            # ---------------------------------------------------------
            self.lbl_status.config(text="⚙️ Generating CV Processing Windows...")
            self.root.update()

            # Pass the save directory to the graphing functions
            self.create_preprocessing_window(image_path, save_dir)
            self.create_morphology_window(image_path, save_dir)
            self.create_features_window(image_path, save_dir)

            self.lbl_status.config(text=f"✅ Done! Outputs saved to: vision_output/{base_filename}/")

        except Exception as e:
            self.lbl_status.config(text="❌ Error during processing.")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Processing Error", str(e))

    # =================================================================
    # REPORT GENERATION METHODS (Opening in new Tkinter Windows & Saving)
    # =================================================================
    
    def display_figure(self, title, fig):
        """Helper to create a new Tkinter window and embed a matplotlib Figure."""
        new_win = tk.Toplevel(self.root)
        new_win.title(title)
        
        # Position windows slightly offset from each other
        offset = len(self.extra_windows) * 30
        new_win.geometry(f"1000x350+{100+offset}+{100+offset}")
        
        self.extra_windows.append(new_win)

        canvas = FigureCanvasTkAgg(fig, master=new_win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_preprocessing_window(self, image_path, save_dir):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        fig = Figure(figsize=(12, 3.5), dpi=100)
        axes = fig.subplots(1, 4)
        
        titles = ['(a) Original Input', '(b) Grayscale Conversion', '(c) Gaussian Blur', '(d) Adaptive Thresholding']
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), gray, blurred, binary]
        cmaps = [None, 'gray', 'gray', 'gray']

        for i in range(4):
            axes[i].imshow(images[i], cmap=cmaps[i])
            axes[i].set_title(titles[i], fontsize=10)
            axes[i].axis('off')
        
        fig.tight_layout()
        
        # --- SAVE THE FIGURE ---
        save_path = os.path.join(save_dir, "1_preprocessing_pipeline.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.display_figure("Report View: Preprocessing Pipeline", fig)

    def create_morphology_window(self, image_path, save_dir):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        fig = Figure(figsize=(10, 3.5), dpi=100)
        axes = fig.subplots(1, 3)
        
        titles = ['(a) Raw Binary', '(b) Dilation', '(c) Morphological Closing']
        images = [binary, dilated, closing]

        for i in range(3):
            axes[i].imshow(images[i], cmap='gray')
            axes[i].set_title(titles[i], fontsize=10)
            axes[i].axis('off')
            
        fig.tight_layout()
        
        # --- SAVE THE FIGURE ---
        save_path = os.path.join(save_dir, "2_morphological_operations.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.display_figure("Report View: Morphological Operations", fig)

    def create_features_window(self, image_path, save_dir):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_boxes = img.copy()
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)

        fig = Figure(figsize=(10, 3.5), dpi=100)
        axes = fig.subplots(1, 3)
        
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('(a) Original Image', fontsize=10)
        axes[1].imshow(edges, cmap='magma')
        axes[1].set_title('(b) Canny Edge Detection', fontsize=10)
        axes[2].imshow(cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB))
        axes[2].set_title('(c) Contour Bounding Boxes', fontsize=10)

        for ax in axes:
            ax.axis('off')
            
        fig.tight_layout()
        
        # --- SAVE THE FIGURE ---
        save_path = os.path.join(save_dir, "3_feature_extraction.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.display_figure("Report View: Feature Extraction", fig)


def main():
    root = tk.Tk()
    try:
        root.state('zoomed')
    except:
        pass
        
    app = VisionTesterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()