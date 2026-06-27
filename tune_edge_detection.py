#/usr/bin/env python3
"""
tune_edge_detection.py

Interactive tuning utility to find the perfect Canny edge detection and morphological closing 
parameters for circuit schematics.

This version uses Tkinter for sliders and image rendering, avoiding headless OpenCV highgui limitations.

Usage:
    python tune_edge_detection.py [path/to/image.png]
"""

import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))


class EdgeTunerApp:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Edge Detection Tuner — Interactive Parameters")
        self.root.configure(bg="#2d2d2d")
        
        self.image_path = image_path
        self.original_cv = cv2.imread(image_path)
        if self.original_cv is None:
            print(f"❌ Failed to load image: {image_path}")
            sys.exit(1)
            
        self.gray_cv = cv2.cvtColor(self.original_cv, cv2.COLOR_BGR2GRAY)
        
        # Initial values
        self.canny_low = 50
        self.canny_high = 150
        self.blur_ksize = 5
        self.morph_ksize = 5
        
        # Build UI layout
        self.setup_ui()
        self.update_images()
        
    def setup_ui(self):
        # 1. Main visual frame (side-by-side images)
        self.display_frame = tk.Frame(self.root, bg="#2d2d2d")
        self.display_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        
        # Left Panel (Original)
        self.left_frame = tk.Frame(self.display_frame, bg="#2d2d2d")
        self.left_frame.pack(side="left", fill="both", expand=True)
        self.lbl_left_title = tk.Label(
            self.left_frame, text="Original Image", bg="#2d2d2d", fg="white", 
            font=("Segoe UI", 10, "bold")
        )
        self.lbl_left_title.pack(anchor="n", pady=(0, 5))
        self.lbl_left_img = tk.Label(self.left_frame, bg="#1e1e1e")
        self.lbl_left_img.pack(expand=True, fill="both")
        
        # Right Panel (Edge Detected)
        self.right_frame = tk.Frame(self.display_frame, bg="#2d2d2d")
        self.right_frame.pack(side="right", fill="both", expand=True)
        self.lbl_right_title = tk.Label(
            self.right_frame, text="Processed Edge Map", bg="#2d2d2d", fg="white", 
            font=("Segoe UI", 10, "bold")
        )
        self.lbl_right_title.pack(anchor="n", pady=(0, 5))
        self.lbl_right_img = tk.Label(self.right_frame, bg="#1e1e1e")
        self.lbl_right_img.pack(expand=True, fill="both")
        
        # 2. Control frame (sliders and buttons)
        self.control_frame = tk.Frame(self.root, bg="#252526", pady=15)
        self.control_frame.pack(side="bottom", fill="x")
        
        # Slider variables
        self.var_low = tk.IntVar(value=self.canny_low)
        self.var_high = tk.IntVar(value=self.canny_high)
        self.var_blur = tk.IntVar(value=self.blur_ksize)
        self.var_morph = tk.IntVar(value=self.morph_ksize)
        
        # Slider layout container
        self.sliders_container = tk.Frame(self.control_frame, bg="#252526")
        self.sliders_container.pack(fill="x", padx=10)
        
        self.create_slider(self.sliders_container, "Canny Low Threshold", self.var_low, 0, 255, 0)
        self.create_slider(self.sliders_container, "Canny High Threshold", self.var_high, 0, 255, 1)
        self.create_slider(self.sliders_container, "Gaussian Blur Size (Odd Only)", self.var_blur, 1, 31, 2, step=2)
        self.create_slider(self.sliders_container, "Morphological Closing Size", self.var_morph, 1, 31, 3)
        
        # Print & Exit Button
        btn_close = tk.Button(
            self.control_frame, text="Accept & Print Parameters", 
            command=self.root.destroy, bg="#0078D7", fg="white", 
            font=("Segoe UI", 10, "bold"), relief="flat", padx=25, pady=8
        )
        btn_close.pack(side="bottom", pady=(15, 0))

    def create_slider(self, parent, label_text, variable, min_val, max_val, col, step=1):
        f = tk.Frame(parent, bg="#252526")
        f.pack(side="left", fill="x", expand=True, padx=10)
        
        lbl = tk.Label(f, text=label_text, bg="#252526", fg="white", font=("Segoe UI", 9))
        lbl.pack(anchor="w")
        
        def on_change(val):
            val_int = int(float(val))
            if step == 2 and val_int % 2 == 0:
                val_int = val_int + 1 if val_int < max_val else val_int - 1
                scale.set(val_int)
            self.update_images()
            
        scale = tk.Scale(
            f, from_=min_val, to=max_val, variable=variable, orient="horizontal",
            bg="#252526", fg="white", highlightthickness=0, resolution=step,
            command=on_change, activebackground="#0078D7"
        )
        scale.pack(fill="x")

    def update_images(self):
        canny_low = self.var_low.get()
        canny_high = self.var_high.get()
        blur_val = self.var_blur.get()
        
        # Ensure blur is odd and >= 1
        blur_ksize = blur_val if blur_val % 2 == 1 else blur_val + 1
        if blur_ksize < 1:
            blur_ksize = 1
            
        morph_ksize = self.var_morph.get()
        if morph_ksize < 1:
            morph_ksize = 1
            
        # Record final values for exit print
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.blur_ksize = blur_ksize
        self.morph_ksize = morph_ksize
        
        # 1. Processing grey image
        if blur_ksize > 1:
            blurred = cv2.GaussianBlur(self.gray_cv, (blur_ksize, blur_ksize), 0)
        else:
            blurred = self.gray_cv.copy()
            
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        if morph_ksize > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize))
            binary = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        else:
            binary = edges.copy()
            
        # 2. Resize and Convert for display (Target height = 500px)
        target_h = 500
        
        # Left Panel (Original BGR)
        h_orig, w_orig = self.original_cv.shape[:2]
        w_new = int(w_orig * (target_h / h_orig))
        
        orig_resized = cv2.resize(self.original_cv, (w_new, target_h), interpolation=cv2.INTER_AREA)
        orig_rgb = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB)
        pil_orig = Image.fromarray(orig_rgb)
        
        # Right Panel (Binary Edge Mask)
        binary_resized = cv2.resize(binary, (w_new, target_h), interpolation=cv2.INTER_AREA)
        pil_bin = Image.fromarray(binary_resized)
        
        # Keep references to images
        self.tk_orig = ImageTk.PhotoImage(pil_orig)
        self.tk_bin = ImageTk.PhotoImage(pil_bin)
        
        # Update labels
        self.lbl_left_img.config(image=self.tk_orig)
        self.lbl_right_img.config(image=self.tk_bin)
        
        # Update title labels/status
        self.lbl_right_title.config(
            text=f"Edges (Canny=[{canny_low}, {canny_high}], Blur={blur_ksize}, Morph={morph_ksize})"
        )


def main():
    # ── 1. Select Input Image ──
    image_path = None
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("📁 Opening file dialog to choose an image...")
        try:
            root = tk.Tk()
            root.withdraw()
            image_path = filedialog.askopenfilename(
                title="Select Image to Tune Edge Detection",
                filetypes=[
                    ("Image Files", "*.png *.jpg *.jpeg *.webp *.bmp *.JPG"),
                    ("All Files", "*.*")
                ]
            )
            root.destroy()
        except Exception as e:
            print(f"⚠️ Could not open file dialog: {e}")

    # Fallback to test images
    if not image_path or not os.path.exists(image_path):
        fallback_dir = os.path.join(WORKSPACE_ROOT, "random_images_for_test")
        if os.path.exists(fallback_dir):
            images = [f for f in os.listdir(fallback_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            if images:
                image_path = os.path.join(fallback_dir, images[0])
                print(f"💡 Defaulted to fallback test image: {image_path}")
            else:
                print("❌ No test images available.")
                sys.exit(1)
        else:
            print("❌ No image selected and no default test folder exists.")
            sys.exit(1)

    print(f"📷 Loading: {image_path}")

    # ── 2. Run Tkinter App ──
    root = tk.Tk()
    app = EdgeTunerApp(root, image_path)
    root.mainloop()

    # ── 3. Print Optimized Output Values ──
    print("\n🎉 Tuning session complete!")
    print("------------------------------------------")
    print(f"Optimal parameters found:")
    print(f"  Canny Low Threshold   : {app.canny_low}")
    print(f"  Canny High Threshold  : {app.canny_high}")
    print(f"  Gaussian Blur Kernel  : {app.blur_ksize} (odd)")
    print(f"  Morphological Closing : {app.morph_ksize}")
    print("------------------------------------------")
    print(f"Update your processing.py or config using:")
    print(f"preprocess_image(file, canny_low={app.canny_low}, canny_high={app.canny_high}, blur_ksize={app.blur_ksize}, morph_ksize={app.morph_ksize})")
    print("------------------------------------------\n")


if __name__ == "__main__":
    main()
