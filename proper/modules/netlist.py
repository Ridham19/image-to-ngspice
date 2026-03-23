import cv2
import numpy as np
from modules.config import cfg
from modules.corrector import run_linter # Import our custom linter

def trace_nodes(wire_mask, detected_components):
    print("🔌 Using OpenCV Pixel-Tracing Engine...")
    h, w = wire_mask.shape
    
    # 1. Clean the mask: Remove text boxes completely so they don't act as accidental wires
    clean_mask = wire_mask.copy()
    for comp in detected_components:
        if comp['type'] == 'text':
            x, y, cw, ch = comp['box']
            cv2.rectangle(clean_mask, (max(0, x-5), max(0, y-5)), (min(w, x+cw+5), min(h, y+ch+5)), 0, -1)

    # 2. "The Core Cut": Erase the middle of components to break short-circuits.
    for comp in detected_components:
        label = comp['type']
        if label in ['junction', 'wire', 'text', 'ground']:
            continue
            
        x, y, cw, ch = comp['box']
        # Cut a massive chunk out of the center, leaving only the edges (pins) touching the wires
        cut_y1, cut_y2 = y + int(ch * 0.2), y + int(ch * 0.8)
        cut_x1, cut_x2 = x + int(cw * 0.2), x + int(cw * 0.8)
        cv2.rectangle(clean_mask, (cut_x1, cut_y1), (cut_x2, cut_y2), 0, -1)

    # 3. Thicken remaining wires and find distinct connected components
    kernel = np.ones((5, 5), np.uint8)
    healed_wires = cv2.dilate(clean_mask, kernel, iterations=1)
    num_labels, labels_im = cv2.connectedComponents(healed_wires)
    
    # 4. Handle Junctions using Union-Find (Merge overlapping nodes!)
    # If a YOLO junction box touches Node 2 and Node 5, they become the same SPICE node.
    parent = {i: i for i in range(num_labels)}
    
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]
        
    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j: parent[root_i] = root_j

    for comp in detected_components:
        if comp['type'] == 'junction':
            x, y, cw, ch = comp['box']
            roi = labels_im[max(0, y-5):min(h, y+ch+5), max(0, x-5):min(w, x+cw+5)]
            valid_nodes = [n for n in np.unique(roi) if n != 0]
            
            # Tie all wires touching this junction together
            for i in range(1, len(valid_nodes)):
                union(valid_nodes[0], valid_nodes[i])
                
    # Apply junction mappings to the master image
    mapped_labels = np.zeros_like(labels_im)
    for i in range(1, num_labels):
        mapped_labels[labels_im == i] = find(i)

    # 5. Assign the distinct nodes back to the components
    comp_connections = []
    
    for comp in detected_components:
        x, y, cw, ch = comp['box']
        label = comp['type']
        
        if label in ['junction', 'wire', 'text']:
            comp_connections.append([])
            continue
            
        pad = 15 # Look just outside the bounding box for wires
        y1, y2 = max(0, y-pad), min(h, y+ch+pad)
        x1, x2 = max(0, x-pad), min(w, x+cw+pad)
        
        roi = mapped_labels[y1:y2, x1:x2]
        valid_nodes = [n for n in np.unique(roi) if n != 0]
        
        # Sort nodes spatially
        node_positions = []
        for node_id in valid_nodes:
            ys, xs = np.where(roi == node_id)
            if len(ys) > 0:
                node_positions.append((node_id, np.mean(xs), np.mean(ys)))
        
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