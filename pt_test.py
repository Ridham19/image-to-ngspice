import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import os

class CircuitAI_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Circuit Component Detector")
        self.root.geometry("1000x800")
        self.root.configure(bg="#2c3e50")

        # 1. Load the Model
        self.model = YOLO('best.pt')

        # 2. UI Elements
        self.label_title = Label(root, text="Circuit AI Detection Studio", font=("Arial", 20, "bold"), fg="white", bg="#2c3e50")
        self.label_title.pack(pady=20)

        self.btn_browse = Button(root, text="📂 Browse Circuit Image", command=self.browse_and_predict, 
                                 font=("Arial", 12), bg="#3498db", fg="white", padx=20, pady=10)
        self.btn_browse.pack()

        # Canvas/Label to display the image
        self.panel = Label(root, bg="#34495e")
        self.panel.pack(pady=20, expand=True)

        self.status = Label(root, text="Ready: Please select an image", font=("Arial", 10), fg="#ecf0f1", bg="#2c3e50")
        self.status.pack(side="bottom", fill="x")

    def browse_and_predict(self):
        # Open File Dialog
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.JPG")])
        
        if file_path:
            self.status.config(text=f"Processing: {os.path.basename(file_path)}...")
            self.root.update_idletasks()

            # 1. Run YOLO Prediction
            # We set stream=True to get the result object directly
            results = self.model.predict(source=file_path, conf=0.3)

            # 2. Get the plotted image (the one with boxes)
            # results[0].plot() returns a BGR numpy array
            res_plotted = results[0].plot()
            
            # Convert BGR to RGB for Tkinter
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            # 3. Resize image to fit the GUI while maintaining aspect ratio
            img_pil = Image.fromarray(res_rgb)
            img_pil.thumbnail((800, 600)) # Maximum size in GUI
            
            # 4. Display in Tkinter
            img_tk = ImageTk.PhotoImage(img_pil)
            self.panel.config(image=img_tk)
            self.panel.image = img_tk  # Keep a reference!
            
            self.status.config(text=f"✅ Detection complete! Found {len(results[0].boxes)} components.")

if __name__ == "__main__":
    root = tk.Tk()
    app = CircuitAI_GUI(root)
    root.mainloop()