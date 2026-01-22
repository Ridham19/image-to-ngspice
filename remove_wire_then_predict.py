import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn

# --- 1. Model Configuration ---
class CircuitNet(nn.Module):
    def __init__(self, num_classes=9): 
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['bjt', 'capacitor', 'diode', 'ground', 'inductor', 'mosfet', 'resistor', 'voltage', 'wire']
spice_prefixes = {'resistor': 'R', 'capacitor': 'C', 'inductor': 'L', 'bjt': 'Q', 'mosfet': 'M', 'diode': 'D', 'voltage': 'V', 'ground': 'GND'}

model = CircuitNet(num_classes=len(class_names)).to(device)
try:
    model.load_state_dict(torch.load("circuit_model.pth", map_location=device))
    model.eval()
    print("Model loaded successfully.")
except:
    print("Error: 'circuit_model.pth' not found.")
    exit()

transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- 2. Intelligent Pre-Processing ---

def clean_image_logic(gray_img, binary_thresh):
    """
    Returns:
    1. display_img: Original image with RED boxes drawn around detected text.
    2. ai_input_img: Grayscale image with text ERASED (filled with white) for the AI.
    3. component_mask: Binary mask with text AND wires removed (for finding component locations).
    """
    display_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    
    # 1. Create a "Clean" Image for AI (Copy of Gray)
    ai_input_img = gray_img.copy()
    
    # 2. Find Small Blobs (Text)
    contours, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cleaned_binary = binary_thresh.copy()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # TEXT FILTER: 
        # Text is usually small area (< 600) OR very squat/tall aspect ratio
        is_text = False
        if area < 600: is_text = True
        
        if is_text:
            # A. Draw Red Box for User
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
            
            # B. ERASE from AI Input (Fill with average background color, usually 255/White)
            # We use 255 (white) effectively erasing the text
            cv2.drawContours(ai_input_img, [cnt], -1, (255), thickness=cv2.FILLED)
            
            # C. ERASE from Binary Mask (so we don't try to predict it later)
            cv2.drawContours(cleaned_binary, [cnt], -1, 0, thickness=cv2.FILLED)

    # 3. Wire Removal (on the image that already has text removed)
    # Remove Horizontal Wires
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    remove_h = cv2.morphologyEx(cleaned_binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
    
    # Remove Vertical Wires
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    remove_v = cv2.morphologyEx(cleaned_binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
    
    # Final Component Mask
    component_mask = cleaned_binary - remove_h - remove_v
    
    # Heal the components (connect broken pieces)
    kernel = np.ones((5,5), np.uint8)
    component_mask = cv2.dilate(component_mask, kernel, iterations=3)
    
    return display_img, ai_input_img, component_mask

# --- 3. Main Analysis ---
def analyze_circuit(image_path):
    img = cv2.imread(image_path)
    if img is None: return print("Image not found")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # --- RUN THE CLEANER ---
    display_img, ai_clean_gray, location_mask = clean_image_logic(gray, thresh)
    
    contours, _ = cv2.findContours(location_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    counts = {name: 0 for name in class_names}
    
    print(f"Scanning for components...")

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = w * h
        
        # Component Filter: Must be decent size and shape
        if area > 500 and 0.2 < aspect_ratio < 5.0:
            
            # Padding
            pad = 20
            y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)
            
            # CRITICAL CHANGE: We crop from 'ai_clean_gray', NOT 'gray'
            # This means the AI sees the component, but the text nearby is ERASED (White).
            roi = ai_clean_gray[y1:y2, x1:x2]
            
            if roi.size == 0: continue
            
            roi_pil = Image.fromarray(roi)
            roi_tensor = transform(roi_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(roi_tensor)
                prob = torch.nn.functional.softmax(output[0], dim=0)
                conf, pred = torch.max(prob, 0)
            
            label = class_names[pred.item()]
            
            # Filter Logic
            if label == 'wire': continue
            
            if conf > 0.65:
                counts[label] += 1
                name = f"{spice_prefixes.get(label, '?')}{counts[label]}"
                if label == 'ground': name = 'GND'
                
                print(f"  -> Found {name} ({label}) Conf: {conf:.2f}")
                
                # Draw Green Box on the display image
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_img, name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    # --- Display Results ---
    # Add border for cleaner look
    final_view = cv2.copyMakeBorder(display_img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    cv2.namedWindow("Final Analysis (Red=Ignored Text)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Final Analysis (Red=Ignored Text)", 1000, 700)
    cv2.imshow("Final Analysis (Red=Ignored Text)", final_view)
    
    # Optional: See what the AI actually analyzed (The "Eraser" view)
    # cv2.imshow("Debug: AI Input (Text Erased)", cv2.resize(ai_clean_gray, (600, 600)))

    print(f"Final Counts: {counts}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    analyze_circuit("DATA/dataset_closed_loop/closed_loop_2comps_0.png")