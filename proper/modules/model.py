import os
import cv2
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Error: 'ultralytics' library missing. Install it: pip install ultralytics")

class ComponentDetector:
    def __init__(self, model_name="best.pt"):
        # Locate model in the 'proper' root folder
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(root_dir, model_name)

        print(f"🧠 Loading Model: {model_path}")
        self.model = None
        
        if os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                print(f"✅ YOLO Model loaded successfully!")
            except Exception as e:
                print(f"❌ Failed to load YOLO model: {e}")
        else:
            print(f"⚠️ Warning: Model file not found at {model_path}")

    def detect(self, image_source):
        """
        Runs YOLO detection.
        Args:
            image_source: Path to image OR numpy array (OpenCV image)
        Returns: 
            List of dicts: {'label': str, 'box': [x, y, w, h], 'conf': float}
        """
        if self.model is None: 
            return []

        # Run inference (conf=0.4 filters weak predictions)
        results = self.model.predict(image_source, conf=0.40, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                # Get Box Coordinates (xyxy)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w, h = x2 - x1, y2 - y1
                
                # Get Class and Confidence
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                conf = float(box.conf[0])
                
                detections.append({
                    'label': label,
                    'box': [x1, y1, w, h],
                    'conf': conf
                })
        
        return detections