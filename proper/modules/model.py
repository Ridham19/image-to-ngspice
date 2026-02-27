import os
import cv2
import math
import json # <--- Import JSON

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Error: 'ultralytics' library missing. Install it: pip install ultralytics")

def calculate_center(box):
    x, y, w, h = box
    return x + (w / 2), y + (h / 2)

class ComponentDetector:
    def __init__(self, model_name="best.pt"):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(root_dir, model_name)

        print(f"🧠 Loading Model: {model_path}")
        self.model = None
        if os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                print(f"❌ Failed to load YOLO model: {e}")

    def detect(self, image_source, output_file="detected_components.json"):
        if self.model is None: 
            return []

        results = self.model.predict(image_source, conf=0.40, verbose=False)
        
        raw_detections = []
        # Generate unique names (e.g., R1, C1)
        counters = {}

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w, h = x2 - x1, y2 - y1
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                conf = float(box.conf[0])
                
                # Assign a unique name like R1, C2, etc.
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
        
        # --- SPATIAL TEXT MATCHING ---
        components = []
        texts = []

        for det in raw_detections:
            if det['type'] == 'text':
                texts.append(det)
            else:
                components.append(det)

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
                comp['value'] = "TEXT_FOUND"
                comp['text_box'] = closest_text['box']

        # 💾 SAVE TO JSON FILE 💾
        print(f"💾 Saving detailed component data to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(components, f, indent=4)

        return components