import argparse
from PIL import Image
from modules.model import ComponentDetector
from modules.config import cfg
from modules.processing import preprocess_image, separate_layers, get_component_contours
from modules.netlist import trace_nodes, generate_spice_text
from modules.gui import show_results
import modules.labeling as labeler
from modules.config import cfg
from modules.gui import run_main_gui


def main_pipeline(image_path):
    print(f"--- Processing {image_path} ---")
    
    # 1. Load & Preprocess
    original, gray, binary = preprocess_image(image_path)
    
    # 2. Separate Layers (Wires vs Components)
    comp_mask, wire_mask, ai_input_gray = separate_layers(gray, binary)
    
    # 3. Detect Components (AI)
    detector = ComponentDetector("best.pt")
    contours = get_component_contours(comp_mask)
    
    detected_comps = []
    counts = {k:0 for k in cfg.class_names}
    
    for (x, y, w, h) in contours:
        # Crop from the "clean" AI input (no text)
        pad = 10
        roi = ai_input_gray[max(0, y-pad):min(gray.shape[0], y+h+pad), 
                            max(0, x-pad):min(gray.shape[1], x+w+pad)]
        
        if roi.size == 0: continue
        
        # Predict
        pil_roi = Image.fromarray(roi)
        label, conf = detector.predict(pil_roi)
        
        if label in ['wire', 'text']: continue
        
        if conf > 0.60:
            counts[label] += 1
            # Get prefix dynamically
            prefix = cfg.specs[label]['prefix']
            name = f"{prefix}{counts[label]}"
            
            detected_comps.append({'name': name, 'type': label, 'box': (x, y, w, h)})
            print(f"Found {name} ({label}) {conf:.2f}")

    # 4. Trace Connectivity
    connections = trace_nodes(wire_mask, detected_comps)
    
    # 5. Generate Netlist
    spice_code = generate_spice_text(detected_comps, connections)
    print("\n" + spice_code + "\n")
    
    # 6. Show GUI
    show_results(original, detected_comps, wire_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", action="store_true", help="Launch manual labeler tool")
    args = parser.parse_args()

    if args.label:
        labeler.run_labeler()
    else:
        # Launch the full main GUI application
        run_main_gui()