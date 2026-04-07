import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import pytesseract
import re

# --- 1. Model Architecture ---
class CircuitNet(nn.Module):
    def __init__(self, num_classes=7):
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

# --- 2. Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ["ac_source", "bjt", "capacitor", "current_source", "diode", "ground", "inductor", "junction", "led", "logic_gate", "mosfet", "opamp", "potentiometer", "resistor", "switch", "text", "transformer", "voltage_source", "wire"]

spice_prefixes = {
    'resistor': 'R', 'capacitor': 'C', 'inductor': 'L', 
    'bjt': 'Q', 'mosfet': 'M', 'voltage': 'V'
}

model = CircuitNet(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("circuit_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(1), transforms.Resize((64, 64)),
    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
])

# -------------------------------
# 🔥 SMART TEXT CLEANING ENGINE
# -------------------------------
def clean_text(text):
    text = text.lower().strip()

    replacements = {
        'o': '0', 'q': '0',
        'l': '1', 'i': '1',
        'z': '2',
        's': '5',
        'b': '6',
        'g': '9'
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    # remove unwanted chars
    text = re.sub(r'[^0-9kmunpfv]', '', text)

    return text


# -------------------------------
# 🔥 VALUE PARSER (SMART)
# -------------------------------
def parse_value(text, comp_type):
    match = re.match(r'(\d+)([kmunpf]*)([a-z]*)', text)
    if not match:
        return None

    num, prefix, unit = match.groups()

    prefix_map = {
        'k': 'k', 'm': 'm', 'u': 'u',
        'n': 'n', 'p': 'p'
    }

    prefix = prefix_map.get(prefix, '')

    if comp_type == 'resistor':
        return f"{num}{prefix}"
    elif comp_type == 'capacitor':
        return f"{num}{prefix}F"
    elif comp_type == 'voltage':
        return f"{num}{prefix}V"

    return f"{num}{prefix}"


# -------------------------------
# 🔥 DISTANCE HELPER
# -------------------------------
def get_center(box):
    x, y, w, h = box
    return (x + w//2, y + h//2)

def distance(b1, b2):
    c1 = get_center(b1)
    c2 = get_center(b2)
    return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5


# -------------------------------
# 🔥 OCR FUNCTION (IMPROVED)
# -------------------------------
def extract_text(roi):
    # upscale
    roi = cv2.resize(roi, None, fx=3, fy=3)

    # blur + threshold
    roi = cv2.GaussianBlur(roi, (3,3), 0)
    _, roi = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)

    config = '--psm 7 -c tessedit_char_whitelist=0123456789kKmMuUnNpPfFVv'
    
    raw = pytesseract.image_to_string(roi, config=config)
    cleaned = clean_text(raw)

    return raw.strip(), cleaned


# -------------------------------
# --- 3. MAIN LOGIC ---
# -------------------------------
def generate_netlist(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    components = []
    text_boxes = []
    counts = {name: 0 for name in class_names}

    print("\n--- Detecting Components ---")

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w > 15 and h > 15:
            roi = gray[y:y+h, x:x+w]
            roi_pil = Image.fromarray(roi)
            roi_tensor = transform(roi_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(roi_tensor)
                prob = torch.nn.functional.softmax(output[0], dim=0)
                conf, pred = torch.max(prob, 0)

            label = class_names[pred.item()]

            # 🔥 Detect text-like regions
            if w < 80 and h < 40:
                text_boxes.append((x, y, w, h))
                continue

            if label == 'wire':
                continue

            if conf > 0.7:
                counts[label] += 1
                name = f"{spice_prefixes[label]}{counts[label]}"

                components.append({
                    'name': name,
                    'type': label,
                    'box': (x, y, w, h),
                    'value': None
                })

                print(f"Detected {name} ({label})")

                cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

    # -------------------------------
    # 🔥 OCR STAGE
    # -------------------------------
    print("\n--- OCR Detection ---")

    extracted_texts = []

    for (x, y, w, h) in text_boxes:
        roi = gray[y:y+h, x:x+w]

        raw, cleaned = extract_text(roi)

        if cleaned:
            print(f"Raw: {raw} → Cleaned: {cleaned}")
            extracted_texts.append((cleaned, (x, y, w, h)))

            cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)

    # -------------------------------
    # 🔥 VALUE MAPPING
    # -------------------------------
    print("\n--- Mapping Values ---")

    for text, tbox in extracted_texts:
        nearest = None
        min_dist = float('inf')

        for comp in components:
            d = distance(tbox, comp['box'])
            if d < min_dist:
                min_dist = d
                nearest = comp

        if nearest:
            parsed = parse_value(text, nearest['type'])

            if parsed:
                nearest['value'] = parsed
            else:
                nearest['value'] = text

            print(f"{text} → {nearest['name']} ({nearest['value']})")

    # -------------------------------
    # 🔥 FINAL NETLIST
    # -------------------------------
    print("\n--- Final Netlist ---")

    for comp in components:
        value = comp['value'] if comp['value'] else "1k"
        print(f"{comp['name']} node1 node2 {value}")

    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    generate_netlist("dataset/images/circuit_2.png")