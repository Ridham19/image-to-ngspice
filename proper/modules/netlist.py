import cv2
import numpy as np
from modules.config import cfg
from modules.corrector import run_linter # Import our custom linter

def trace_nodes(wire_mask, detected_components):
    h, w = wire_mask.shape
    
    # 1. "The Core Cut": Erase the middle of components to break short-circuits.
    cut_wires = wire_mask.copy()
    
    for comp in detected_components:
        label = comp['type']
        if label in ['junction', 'wire', 'text']:
            continue
            
        x, y, cw, ch = comp['box']
        
        if ch >= cw: 
            cut_y1 = y + (ch // 2) - 4
            cut_y2 = y + (ch // 2) + 4
            cv2.rectangle(cut_wires, (max(0, x-5), cut_y1), (min(w, x+cw+5), cut_y2), 0, -1)
        else:
            cut_x1 = x + (cw // 2) - 4
            cut_x2 = x + (cw // 2) + 4
            cv2.rectangle(cut_wires, (cut_x1, max(0, y-5)), (cut_x2, min(h, y+ch+5)), 0, -1)

    # 2. Heal the remaining true wires and find distinct connected components
    kernel = np.ones((5, 5), np.uint8)
    healed_wires = cv2.dilate(cut_wires, kernel, iterations=1)
    
    num_labels, labels_im = cv2.connectedComponents(healed_wires)
    comp_connections = []
    
    # 3. Assign the distinct nodes to the components
    for comp in detected_components:
        x, y, cw, ch = comp['box']
        label = comp['type']
        
        if label in ['junction', 'wire', 'text']:
            comp_connections.append([])
            continue
            
        pad = 12
        y1, y2 = max(0, y-pad), min(h, y+ch+pad)
        x1, x2 = max(0, x-pad), min(w, x+cw+pad)
        
        roi = labels_im[y1:y2, x1:x2]
        
        unique_nodes = np.unique(roi)
        valid_nodes = [n for n in unique_nodes if n != 0]
        
        # 4. Sort nodes spatially
        node_positions = []
        for node_id in valid_nodes:
            ys, xs = np.where(roi == node_id)
            if len(ys) > 0:
                avg_y = np.mean(ys)
                avg_x = np.mean(xs)
                node_positions.append((node_id, avg_x, avg_y))
        
        is_vertical = ch >= cw
        if is_vertical:
            node_positions.sort(key=lambda item: item[2]) # Top to Bottom
        else:
            node_positions.sort(key=lambda item: item[1]) # Left to Right
            
        sorted_nodes = [item[0] for item in node_positions]
        comp_connections.append(sorted_nodes)

    return comp_connections

def generate_spice_text(components, connections):
    netlist = ["* Auto-Generated SPICE Netlist"]
    
    # 1. Build Raw Component Lines
    for i, comp in enumerate(components):
        label = comp['type']
        name = comp['name']
        nodes = connections[i]
        
        # Skip utility symbols in the netlist output
        if label in ['ground', 'wire', 'text', 'junction']:
            if label == 'ground' and nodes:
                netlist.append(f"* Ground detected at Node {nodes[0]}")
            continue

        expected_pins = cfg.get_pin_names(label)
        if not expected_pins:
            expected_pins = ['p1', 'p2'] 
            
        current_nodes = []
        for j in range(len(expected_pins)):
            if j < len(nodes):
                current_nodes.append(str(nodes[j]))
            else:
                current_nodes.append(f"NC_{name}_{j+1}") # NC = Not Connected

        node_str = " ".join(current_nodes)
        
        # Fallback Defaults
        val = "1k" 
        if label in ['voltage', 'source', 'battery']: val = "5V"
        elif label == 'capacitor': val = "1u"
        elif label == 'inductor': val = "1mH"
        elif label == 'diode': val = "D1"

        # --- NEW: APPLY OCR VALUE ---
        detected_val = comp.get('value')
        if detected_val and detected_val != "TEXT_FOUND":
            val = detected_val # Use the OCR value (e.g., "10k")
        elif detected_val == "TEXT_FOUND":
            val = f"{val}_(NEEDS_OCR)" # Fallback if OCR was too blurry

        netlist.append(f"{name} {node_str} {val}")
        
    raw_netlist_str = "\n".join(netlist)
    
    # 2. Run the Linter automatically before returning
    final_netlist = run_linter(raw_netlist_str)
    
    return final_netlist