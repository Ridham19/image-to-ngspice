import os
import cv2
import math
import json
import re

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Error: 'ultralytics' missing. pip install ultralytics")

try:
    import easyocr
except ImportError:
    print("❌ Error: 'easyocr' missing. pip install easyocr")

def calculate_center(box):
    x, y, w, h = box
    return x + (w / 2), y + (h / 2)

class ComponentDetector:
    def __init__(self, model_name="best.pt"):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(root_dir, model_name)

        print(f"🧠 Loading YOLO Model: {model_path}")
        self.model = None
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            
        print("👁️ Loading EasyOCR Engine (This takes a few seconds)...")
        # Initialize OCR once so it doesn't slow down every image detection
        self.ocr_reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if you have an Nvidia GPU!

    def detect(self, image_source, output_file="detected_components.json"):
        if self.model is None: 
            return []

        results = self.model.predict(image_source, conf=0.40, verbose=False)
        
        # Load the original image so we can crop the text boxes out of it
        if isinstance(image_source, str):
            cv_img = cv2.imread(image_source)
        else:
            cv_img = image_source

        raw_detections = []
        counters = {}

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w, h = x2 - x1, y2 - y1
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                conf = float(box.conf[0])
                
                prefix = label[0].upper() if label not in ['voltage', 'source'] else 'V'
                if label == 'ground': prefix = 'GND'
                
                counters[prefix] = counters.get(prefix, 0) + 1
                name = f"{prefix}{counters[prefix]}"

                raw_detections.append({
                    'name': name,
                    'type': label,
                    'box': [x1, y1, w, h],
                    'center': calculate_center([x1, y1, w, h]),
                    'conf': round(conf, 2),
                    'value': None
                })
        
        # --- SPATIAL TEXT MATCHING & OCR ---
        components = [d for d in raw_detections if d['type'] != 'text']
        texts = [d for d in raw_detections if d['type'] == 'text']

        for comp in components:
            if comp['type'] in ['wire', 'junction', 'ground']:
                continue
                
            comp_center = comp['center']
            closest_text = None
            min_dist = float('inf')

            for text in texts:
                dist = math.dist(comp_center, text['center'])
                if dist < min_dist:
                    min_dist = dist
                    closest_text = text

            if closest_text and min_dist < 150: 
                # 1. We found a text box! Crop it with MORE padding.
                tx, ty, tw, th = closest_text['box']
                pad = 12 # Increased padding
                
                # Safe crop bounds so we don't go outside the image
                h_img, w_img = cv_img.shape[:2]
                y1, y2 = max(0, ty-pad), min(h_img, ty+th+pad)
                x1, x2 = max(0, tx-pad), min(w_img, tx+tw+pad)
                
                crop_img = cv_img[y1:y2, x1:x2]
                
                # 2. ENLARGE the image by 200% so EasyOCR can read handwriting better
                if crop_img.shape[0] > 0 and crop_img.shape[1] > 0:
                    crop_img = cv2.resize(crop_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                
                # 3. Pass the enlarged image to EasyOCR
                ocr_result = self.ocr_reader.readtext(crop_img, detail=0)
                
                if ocr_result:
                    raw_string = "".join(ocr_result)
                    
                    # Clean the text (keep numbers, decimal points, and unit letters)
                    clean_string = re.sub(r'[^a-zA-Z0-9\.]', '', raw_string)
                    
                    # --- THE FIX: HANDWRITING AUTOCORRECT ---
                    # Common OCR handwriting mistakes for circuit values
                    clean_string = clean_string.replace('i', '1').replace('l', '1')
                    clean_string = clean_string.replace('o', '0').replace('O', '0')
                    clean_string = clean_string.replace('S', '5')
                    
                    comp['value'] = clean_string if clean_string else "TEXT_FOUND"
                else:
                    comp['value'] = "TEXT_FOUND"
                    
                comp['text_box'] = closest_text['box']

        # 💾 SAVE TO JSON FILE 💾
        print(f"💾 Saving detailed component data to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(components, f, indent=4)

        return components