import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import os

class SourcePipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Source Classification Tester")
        self.root.geometry("1100x850")
        self.root.configure(bg="#2c3e50")

        # 1. Load the Models
        self.status_var = tk.StringVar(value="⏳ Loading YOLO Models...")
        self.status = Label(root, textvariable=self.status_var, font=("Arial", 11), fg="#ecf0f1", bg="#34495e", anchor="w", padx=10)
        self.status.pack(side="bottom", fill="x")
        self.root.update()

        try:
            self.detector = YOLO('best.pt')
            self.classifier = YOLO('source_classifier_massive.pt')
            self.status_var.set("✅ Models loaded successfully. Ready to process images.")
        except Exception as e:
            self.status_var.set(f"❌ Error loading models: {str(e)}")

        # 2. UI Title Elements
        self.label_title = Label(root, text="Circuit Source Classification Studio", font=("Arial", 22, "bold"), fg="white", bg="#2c3e50")
        self.label_title.pack(pady=15)

        # Control Panel Frame
        self.control_frame = Frame(root, bg="#2c3e50")
        self.control_frame.pack(pady=5)

        self.btn_browse = Button(self.control_frame, text="📂 Browse Circuit Image", command=self.browse_and_predict, 
                                 font=("Arial", 12, "bold"), bg="#3498db", fg="white", padx=20, pady=8, activebackground="#2980b9", activeforeground="white")
        self.btn_browse.grid(row=0, column=0, padx=10)

        # Canvas/Label to display the image
        self.panel = Label(root, bg="#34495e", bd=2, relief="sunken")
        self.panel.pack(pady=15, expand=True, fill="both", padx=20)

    def browse_and_predict(self):
        # Open File Dialog
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.webp *.JPG")])
        
        if file_path:
            self.status_var.set(f"Processing: {os.path.basename(file_path)}...")
            self.root.update_idletasks()

            # Read original image
            img = cv2.imread(file_path)
            if img is None:
                self.status_var.set("❌ Error: Could not read selected image.")
                return

            # Run detection using best.pt
            detect_results = self.detector.predict(img, conf=0.25, verbose=False)
            
            output_img = img.copy()
            source_count = 0
            other_count = 0

            for result in detect_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls_id = int(box.cls[0])
                    label = self.detector.names[cls_id]
                    conf = float(box.conf[0])

                    # Identify source components (label is "source")
                    if label.lower() == 'source':
                        source_count += 1
                        
                        # Crop the source ROI
                        h_img, w_img = img.shape[:2]
                        y1_crop, y2_crop = max(0, y1), min(h_img, y2)
                        x1_crop, x2_crop = max(0, x1), min(w_img, x2)
                        roi = img[y1_crop:y2_crop, x1_crop:x2_crop]
                        
                        classified_label = "Unknown Source"
                        classified_conf = 0.0
                        
                        if roi.size > 0:
                            # Run classification on cropped source
                            class_results = self.classifier.predict(roi, verbose=False)
                            for c_res in class_results:
                                if c_res.boxes is not None and len(c_res.boxes) > 0:
                                    best_box = None
                                    best_conf = -1.0
                                    for b_box in c_res.boxes:
                                        b_conf = float(b_box.conf[0])
                                        if b_conf > best_conf:
                                            best_conf = b_conf
                                            best_box = b_box
                                    
                                    if best_box is not None:
                                        new_cls_id = int(best_box.cls[0])
                                        classified_label = self.classifier.names[new_cls_id]
                                        classified_conf = best_conf
                                elif hasattr(c_res, 'probs') and c_res.probs is not None:
                                    new_cls_id = int(c_res.probs.top1)
                                    classified_label = self.classifier.names[new_cls_id]
                                    classified_conf = float(c_res.probs.top1conf)
                        
                        # Draw high-visibility box (Red) and label for source
                        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        display_text = f"{classified_label} ({classified_conf:.2f})"
                        cv2.putText(output_img, display_text, (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        other_count += 1
                        # Draw other components in green/cyan
                        cv2.rectangle(output_img, (x1, y1), (x2, y2), (46, 204, 113), 2)
                        display_text = f"{label} ({conf:.2f})"
                        cv2.putText(output_img, display_text, (x1, y1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 204, 113), 2)

            # Convert BGR to RGB for PIL / Tkinter
            res_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            
            # Resize image to fit panel nicely
            img_pil = Image.fromarray(res_rgb)
            # Find best fitting size for visual panel area
            panel_width = self.panel.winfo_width()
            panel_height = self.panel.winfo_height()
            if panel_width < 100 or panel_height < 100:
                panel_width, panel_height = 900, 600
                
            img_pil.thumbnail((panel_width, panel_height))
            
            # Display image
            img_tk = ImageTk.PhotoImage(img_pil)
            self.panel.config(image=img_tk)
            self.panel.image = img_tk  # Keep reference
            
            self.status_var.set(f"✅ Detection complete! Found {source_count} source(s) and {other_count} other component(s).")

if __name__ == "__main__":
    root = tk.Tk()
    app = SourcePipelineGUI(root)
    root.mainloop()
