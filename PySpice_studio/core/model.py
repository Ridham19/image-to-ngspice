import os
import cv2
import math
import json
import re
import subprocess
import sys

HAS_ML = True
try:
    from ultralytics import YOLO
    import easyocr
    print("✅ All dependencies are ready.")
except ImportError as e:
    print(f"⚠️ ML dependencies missing or broken: {e}. Running in MOCK mode.")
    HAS_ML = False


def calculate_center(box):
    x, y, w, h = box
    return x + (w / 2), y + (h / 2)

def clean_ocr_string(raw_text):
    """Minimal sanitization of raw OCR text — preserves letters for classification.

    Only lowercases, strips whitespace, normalises the Greek micro symbol to 'u',
    and removes characters that are clearly noise.  Destructive digit-swaps
    (s→5, o→0 etc.) are deliberately NOT applied here — they are deferred to
    _fix_handwriting_digits() which runs only on value strings.
    """
    text = raw_text.strip()
    # Normalise Greek mu (μ / µ) to ASCII 'u' before lowercasing
    text = text.replace('\u03bc', 'u').replace('\u00b5', 'u')
    text = text.lower()
    # Strip trailing omega / ohms indicators (only affects value strings, safe here)
    text = re.sub(r'(ohms?|omega|\u03a9|\u2126)$', '', text)
    # Map trailing 'x' to 'k' (common misread for handwriting 'k' after a number)
    text = re.sub(r'(\d+\.?\d*)x$', r'\1k', text)
    # Keep only alphanumeric, dots, underscores and spaces (spaces needed for composite splitting)
    text = re.sub(r'[^a-z0-9\._\s]', '', text)
    return text.strip()


def _fix_handwriting_digits(text):
    """Apply handwriting-specific digit corrections to a VALUE string only.

    These replacements would corrupt component names (e.g. 'source' → '50urce')
    so they must never be applied to raw / unclassified text.
    """
    text = text.replace('i', '1').replace('l', '1').replace('|', '1')
    text = text.replace('s', '5')
    text = text.replace('z', '2')
    text = text.replace('o', '0')
    return text


# Regex that matches a SPICE-style component name: letter prefix + digits (e.g. R1, C12, V_IN)
_NAME_RE = re.compile(r'^([a-z]{1,3}\d{1,4})$')
# Regex that matches a numeric value with optional unit suffix (e.g. 10k, 4.7u, 0.1meg)
_VALUE_RE = re.compile(r'^(\d+\.?\d*)(k|m|meg|g|u|p|n|f|r|v|a|\w{0,3})?$')
# Regex for composite "name value" blocks (e.g. "R1 10k", "C2 100u")
_COMPOSITE_RE = re.compile(r'^([a-z]{1,3}\d{1,4})\s+(\d[\d\.]*\w{0,3})$')


def classify_text(text, comp_type, comp_prefix):
    """Classify cleaned OCR text as a component 'name', 'value', or 'composite'.

    Returns
    -------
    str  – 'name' or 'value'
    dict – {'name': str, 'value': str}  when a composite block is detected
    None – if the text is empty or unclassifiable
    """
    if not text:
        return None

    # --- 1. Try composite split first (e.g. "r1 10k") ---
    composite_match = _COMPOSITE_RE.match(text)
    if composite_match:
        return {
            'name': composite_match.group(1),
            'value': composite_match.group(2),
        }

    # Remove internal whitespace for single-token classification
    text = re.sub(r'\s+', '', text)
    if not text:
        return None

    has_digit = any(c.isdigit() for c in text)
    starts_with_prefix = comp_prefix and text.startswith(comp_prefix.lower())

    if not has_digit:
        return 'name'

    starts_with_digit = text[0].isdigit() or text[0] == '.'
    if starts_with_digit:
        return 'value'

    if starts_with_prefix:
        return 'name'
    return 'value'

try:
    from core.ocr_validator import correct_ocr_value
except ImportError:
    try:
        from ocr_validator import correct_ocr_value
    except ImportError:
        correct_ocr_value = None

def correct_component_value(value_str, comp_type):
    """Normalise an OCR-read value string into a valid SPICE literal.

    Delegates validation and autocorrection to core.ocr_validator module.
    """
    if not value_str:
        return value_str

    if correct_ocr_value is not None:
        corrected, _, _ = correct_ocr_value(value_str, comp_type)
        return corrected

    comp_type = comp_type.lower()

    # --- Step 0: Protect known unit suffixes from handwriting digit corruption ---
    # Extract trailing unit keywords BEFORE running _fix_handwriting_digits so that
    # e.g. 'ohm' doesn't become '0hm' and 'meg' doesn't become 'meg' -> '1eg' etc.
    preserved_unit = ''
    value_lower = value_str.lower()
    for suffix in ('ohms', 'ohm', 'meg', 'kohm', 'kohms'):
        if value_lower.endswith(suffix):
            preserved_unit = suffix
            value_str = value_str[:len(value_str) - len(suffix)]
            break

    # Apply handwriting digit corrections to the numeric portion only
    value_str = _fix_handwriting_digits(value_str) + preserved_unit

    # --- Step 1: European schematic notation (e.g. 4k7 → 4.7k, 0R1 → 0.1) ---
    internal_unit_match = re.match(r'^(\d+)([krmgupn])(\d+)$', value_str)
    if internal_unit_match:
        num1, unit, num2 = internal_unit_match.groups()
        if unit == 'r':
            # 'r' in European notation means the decimal point sits here (Ohms)
            value_str = f"{num1}.{num2}"
        else:
            value_str = f"{num1}.{num2}{unit}"

    # --- Step 2: Split into numeric part + unit suffix ---
    match = re.match(r'^([\d\.]+)([a-zA-Z]*)$', value_str)
    if not match:
        return value_str

    numeric_part, unit_part = match.groups()

    # Ensure at most one decimal point
    if numeric_part.count('.') > 1:
        parts = numeric_part.split('.')
        numeric_part = parts[0] + '.' + ''.join(parts[1:])

    unit_part = unit_part.lower()

    # --- Step 3: Component-specific unit autocorrect ---
    if 'resistor' in comp_type:
        if unit_part in ('r', 'ohm', 'ohms'):
            # 'r' / 'ohm' means plain Ohms — no SPICE suffix needed
            unit_part = ''
        elif unit_part in ('k', 'x', 'q'):
            unit_part = 'k'
        elif unit_part in ('meg', 'mg'):
            unit_part = 'meg'
        elif unit_part == 'm':
            # For resistors, bare 'm' almost always means Mega, not milli
            unit_part = 'meg'
        elif unit_part == 'g':
            unit_part = 'g'
        elif unit_part:
            # Truly unrecognised unit — default to k as a last resort
            unit_part = 'k'

    elif 'capacitor' in comp_type:
        if unit_part == 'p':
            unit_part = 'p'
        elif unit_part == 'n':
            unit_part = 'n'
        elif unit_part in ('u', 'x'):
            unit_part = 'u'
        elif unit_part == 'm':
            # For capacitors, bare 'm' means milli (mF) — rare but valid
            unit_part = 'm'
        elif unit_part == 'f':
            # Bare 'f' usually means micro-farads in hand-drawn schematics
            unit_part = 'u'
        elif unit_part:
            unit_part = 'u'

    elif 'inductor' in comp_type:
        if unit_part in ('u', 'x'):
            unit_part = 'u'
        elif unit_part == 'n':
            unit_part = 'n'
        elif unit_part == 'm':
            unit_part = 'm'
        elif unit_part == 'h':
            # Bare 'h' (Henries) — no SPICE suffix
            unit_part = ''
        elif unit_part:
            unit_part = 'u'

    elif 'source' in comp_type:
        if 'current' in comp_type:
            unit_part = 'a' if unit_part else ''
        else:
            unit_part = 'v' if unit_part else ''

    return f"{numeric_part}{unit_part}"

def _find_model_file(filename, start_dir):
    if not filename:
        return None
    if os.path.exists(filename):
        return filename
    candidates = [
        os.path.join(start_dir, filename),
        os.path.join(os.path.dirname(start_dir), filename),
        os.path.join(os.path.dirname(os.path.dirname(start_dir)), filename),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None

def _load_config(start_dir):
    config_file = _find_model_file("config.json", start_dir)
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                return json.load(f), os.path.dirname(config_file)
        except Exception as e:
            print(f"⚠️ Could not load config.json: {e}")
    return {}, start_dir

class ComponentDetector:
    def __init__(self, model_name=None):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg, cfg_dir = _load_config(root_dir)

        # Configurable model paths from config.json (with fallback search)
        cfg_yolo = cfg.get("yolo_model_path") or cfg.get("yolo_model")
        cfg_bjt = cfg.get("bjt_model_path") or cfg.get("bjt_model")

        yolo_target = model_name or cfg_yolo or "epoch_40.pt"
        model_path = _find_model_file(yolo_target, root_dir) or _find_model_file(yolo_target, cfg_dir) or _find_model_file("best.pt", root_dir)

        print(f"🧠 Loading Primary YOLO Model: {model_path}")
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)

        bjt_target = cfg_bjt or "bjt_best_model.pt"
        bjt_model_path = _find_model_file(bjt_target, root_dir) or _find_model_file(bjt_target, cfg_dir)

        self.bjt_classifier = None
        if bjt_model_path and os.path.exists(bjt_model_path) and HAS_ML:
            print(f"🧠 Loading BJT Classifier Model: {bjt_model_path}")
            try:
                self.bjt_classifier = YOLO(bjt_model_path)
            except Exception as e:
                print(f"⚠️ Failed to load BJT classifier: {e}")
            
        self.ocr_reader = None
        if HAS_ML:
            print("👁️ Loading EasyOCR Engine (This takes a few seconds)...")
            # Initialize OCR once so it doesn't slow down every image detection
            self.ocr_reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if you have an Nvidia GPU!

    # ═══════════════════════════════════════════
    # CLASS REMAPPING — epoch_40.pt model labels → canvas types
    # ═══════════════════════════════════════════
    CLASS_REMAP = {
        # Direct mappings (dot-notation → underscore canvas type)
        'gnd':                    'ground',
        'vss':                    'vss',
        'voltage.dc':             'source',
        'voltage.ac':             'ac_source',
        'voltage.battery':        'source',
        'resistor.photo':         'resistor_photo',
        'capacitor.unpolarized':  'capacitor',
        'capacitor.polarized':    'capacitor_polarized',
        'diode.light_emitting':   'diode_led',
        'diode.zener':            'diode_zener',
        'transistor.bjt':         'bjt',
        'transistor.fet':         'mosfet',
        'transistor.photo':       'phototransistor',
        'operational_amplifier':  'opamp',
        'integrated_circuit':     'ic',
        # Legacy remap & BJT classifier remap
        'transistor':             'bjt',
        'BjtNpn':                 'bjt_npn',
        'BjtPnp':                 'bjt_pnp',
        'bjt_npn':                'bjt_npn',
        'bjt_pnp':                'bjt_pnp',
    }

    # SPICE prefix for auto-naming
    PREFIX_MAP = {
        'resistor': 'R', 'resistor_photo': 'R',
        'capacitor': 'C', 'capacitor_polarized': 'C',
        'inductor': 'L',
        'diode': 'D', 'diode_led': 'D', 'diode_zener': 'D',
        'source': 'V', 'voltage_source': 'V', 'ac_source': 'V',
        'current_source': 'I',
        'ground': 'GND', 'vss': 'VSS',
        'bjt': 'Q', 'bjt_npn': 'Q', 'bjt_pnp': 'Q',
        'mosfet': 'M', 'phototransistor': 'Q',
        'opamp': 'U', 'ic': 'U',
        'transformer': 'T',
        'junction': 'J', 'crossover': 'X', 'terminal': 'P',
    }

    def detect(self, image_source, output_file="detected_components.json"):
        if not HAS_ML or self.model is None:
            print("⚠️ Returning mock detections because ML libraries are unavailable.")
            return [
                {"name": "R1", "type": "resistor", "box": [100, 100, 80, 40], "center": [140, 120], "conf": 0.99, "value": "1k", "rotation": 0},
                {"name": "V1", "type": "source", "box": [20, 100, 50, 80], "center": [45, 140], "conf": 0.99, "value": "5V", "rotation": 270},
                {"name": "GND1", "type": "ground", "box": [25, 200, 40, 30], "center": [45, 215], "conf": 0.99, "value": None, "rotation": 0}
            ]

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
                raw_label = self.model.names[cls_id]
                conf = float(box.conf[0])
                
                # Remap model class label to canvas type
                label = self.CLASS_REMAP.get(raw_label, raw_label)
                
                prefix = self.PREFIX_MAP.get(label, label[0].upper() if label else 'U')
                
                counters[prefix] = counters.get(prefix, 0) + 1
                name = f"{prefix}{counters[prefix]}"

                # Determine rotation from bounding box aspect ratio
                # If vertical (h > w), rotate 90 degrees anticlockwise (270 degrees in canvas space)
                # Else horizontal, keep rotation as 0 degrees
                rotation = 270 if h > w else 0

                raw_detections.append({
                    'name': name,
                    'type': label,
                    'box': [x1, y1, w, h],
                    'center': calculate_center([x1, y1, w, h]),
                    'conf': round(conf, 2),
                    'value': None,
                    'rotation': rotation
                })

        # --- BJT NPN / PNP CLASSIFICATION REFINEMENT ---
        if self.bjt_classifier is not None and cv_img is not None:
            for comp in raw_detections:
                if comp['type'] in ('bjt', 'transistor', 'bjt_npn', 'bjt_pnp'):
                    x, y, w, h = comp['box']
                    pad_w = int(w * 0.15)
                    pad_h = int(h * 0.15)
                    h_img, w_img = cv_img.shape[:2]
                    x1 = max(0, x - pad_w)
                    y1 = max(0, y - pad_h)
                    x2 = min(w_img, x + w + pad_w)
                    y2 = min(h_img, y + h + pad_h)

                    crop_img = cv_img[y1:y2, x1:x2]
                    if crop_img.shape[0] > 0 and crop_img.shape[1] > 0:
                        try:
                            bjt_results = self.bjt_classifier.predict(crop_img, conf=0.10, verbose=False)
                            best_bjt_cls = None
                            best_bjt_conf = 0.0
                            for res in bjt_results:
                                for b in res.boxes:
                                    c_id = int(b.cls[0])
                                    c_conf = float(b.conf[0])
                                    if c_conf > best_bjt_conf:
                                        best_bjt_conf = c_conf
                                        raw_bjt_label = self.bjt_classifier.names[c_id]
                                        best_bjt_cls = self.CLASS_REMAP.get(raw_bjt_label, 'bjt_npn')

                            if best_bjt_cls:
                                print(f"🔬 BJT Classifier refined {comp['name']} ({comp['type']}) -> {best_bjt_cls} (conf: {best_bjt_conf:.2f})")
                                comp['type'] = best_bjt_cls
                            else:
                                comp['type'] = 'bjt_npn'
                        except Exception as bjt_err:
                            print(f"⚠️ Error running BJT classifier on crop: {bjt_err}")
        
        # --- SPATIAL TEXT MATCHING & OCR ---
        # Junction detections are kept but separated — they act as wire merge points
        NON_OCR_TYPES = {'wire', 'junction', 'crossover', 'terminal', 'ground', 'vss'}
        components = [d for d in raw_detections if d['type'] != 'text']
        texts = [d for d in raw_detections if d['type'] == 'text']

        # Keep track of assigned names to ensure uniqueness
        assigned_names = {comp['name'] for comp in components}

        for comp in components:
            if comp['type'] in NON_OCR_TYPES:
                continue
                
            comp_center = comp['center']
            comp_prefix = self.PREFIX_MAP.get(comp['type'], '')

            # Find all text boxes within 150 pixels of this component
            nearby_texts = []
            for text in texts:
                dist = math.dist(comp_center, text['center'])
                if dist < 150:
                    nearby_texts.append((text, dist))
            
            # Sort by distance
            nearby_texts.sort(key=lambda x: x[1])

            detected_value = None
            detected_name = None
            best_text_box = None

            for text_item, dist in nearby_texts:
                tx, ty, tw, th = text_item['box']
                # Resolution-aware padding: 15% of the larger dimension, clamped [5, 30]
                pad = max(5, min(30, int(max(tw, th) * 0.15)))
                
                # Safe crop bounds so we don't go outside the image
                h_img, w_img = cv_img.shape[:2]
                y1, y2 = max(0, ty-pad), min(h_img, ty+th+pad)
                x1, x2 = max(0, tx-pad), min(w_img, tx+tw+pad)
                
                crop_img = cv_img[y1:y2, x1:x2]
                
                if crop_img.shape[0] > 0 and crop_img.shape[1] > 0:
                    # Enlarge the image based on size (4x for tiny crops under 2500 px², 2x for larger)
                    crop_area = crop_img.shape[0] * crop_img.shape[1]
                    scale = 4.0 if crop_area < 2500 else 2.0
                    resized_crop = cv2.resize(crop_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    
                    # Pass the enlarged image to EasyOCR
                    ocr_result = self.ocr_reader.readtext(resized_crop, detail=0)
                    
                    if ocr_result:
                        raw_string = " ".join(ocr_result)
                        cleaned_str = clean_ocr_string(raw_string)
                        classification = classify_text(cleaned_str, comp['type'], comp_prefix)
                        
                        # Handle composite result (dict with both name + value)
                        if isinstance(classification, dict):
                            if not detected_name:
                                detected_name = classification['name'].upper()
                            if not detected_value:
                                detected_value = correct_component_value(
                                    classification['value'], comp['type']
                                )
                                best_text_box = text_item['box']
                        elif classification == 'value' and not detected_value:
                            # Strip spaces before fixing digits in value path
                            value_str = re.sub(r'\s+', '', cleaned_str)
                            detected_value = correct_component_value(value_str, comp['type'])
                            best_text_box = text_item['box']
                        elif classification == 'name' and not detected_name:
                            detected_name = re.sub(r'\s+', '', cleaned_str).upper()

            if detected_value:
                comp['value'] = detected_value
                comp['text_box'] = best_text_box
            elif len(nearby_texts) > 0:
                comp['value'] = "TEXT_FOUND"

            if detected_name:
                # simple validation: check if the parsed name starts with the expected prefix
                # and satisfies basic length / format constraints
                detected_name_upper = detected_name.upper()
                if comp_prefix and detected_name_upper.startswith(comp_prefix.upper()) and len(detected_name_upper) <= 6:
                    if detected_name_upper not in assigned_names:
                        assigned_names.discard(comp['name'])
                        comp['name'] = detected_name_upper
                        assigned_names.add(detected_name_upper)

        # 💾 SAVE TO JSON FILE 💾
        print(f"💾 Saving detailed component data to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(components, f, indent=4)

        return components