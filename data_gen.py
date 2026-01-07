import os
import schemdraw
import schemdraw.elements as elm
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2

# Configuration
CLASSES = {
    'resistor': elm.Resistor,
    'capacitor': elm.Capacitor,
    'inductor': elm.Inductor,
    'bjt': elm.BjtNpn,
    'mosfet': elm.NFet, 
}

SAMPLES_PER_TYPE = 50  # 50 clean + 50 noisy = 100 total per class
DATASET_ROOT = "dataset_v2"

def apply_noise(image_path):
    """Applies subtle Gaussian noise and slight blurring to simulate a real scan."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return
    
    # 1. Add Gaussian Noise
    gauss = np.random.normal(0, 7, img.shape).astype('uint8')
    img = cv2.add(img, gauss)
    
    # 2. Slight Blur (simulates a non-perfect camera focus)
    if random.choice([True, False]):
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
    cv2.imwrite(image_path, img)

def generate_component(ComponentClass, save_path, is_noisy=False):
    """Draws the component and optionally adds noise."""
    unit_size = random.uniform(2.5, 3.5)
    line_width = random.uniform(1.5, 2.5) 
    
    # Draw
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=unit_size, lw=line_width)
        orientation = random.choice(['right', 'up', 'down', 'left'])
        
        comp = ComponentClass()
        if orientation == 'up': comp.up()
        elif orientation == 'down': comp.down()
        elif orientation == 'left': comp.left()
        else: comp.right()
        
        d.add(comp)
        d.save(save_path, dpi=100)
        
    plt.close('all') 
    
    if is_noisy:
        apply_noise(save_path)

if __name__ == "__main__":
    os.makedirs(DATASET_ROOT, exist_ok=True)
    
    for class_name, comp_obj in CLASSES.items():
        path = os.path.join(DATASET_ROOT, class_name)
        os.makedirs(path, exist_ok=True)
        
        print(f"Generating data for {class_name}...")
        
        # 1. Generate Clean Images
        for i in range(SAMPLES_PER_TYPE):
            filename = os.path.join(path, f"{class_name}_{i}_clean.png")
            generate_component(comp_obj, filename, is_noisy=False)
            
        # 2. Generate Noisy Images
        for i in range(SAMPLES_PER_TYPE):
            filename = os.path.join(path, f"{class_name}_{i}_noisy.png")
            generate_component(comp_obj, filename, is_noisy=True)

    print(f"\nSuccess! Dataset created in '{DATASET_ROOT}' with 100 images per class (50 clean/50 noisy).")