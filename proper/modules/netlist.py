import cv2
import numpy as np
from modules.config import cfg

def trace_nodes(wire_mask, detected_components):
    h, w = wire_mask.shape
    # Dilate wires to ensure they touch components
    dilated_wires = cv2.dilate(wire_mask, np.ones((5,5), np.uint8), iterations=1)
    
    # Each connected wire blob is a "Node"
    num_labels, labels_im = cv2.connectedComponents(dilated_wires)
    comp_connections = [] 
    
    for comp in detected_components:
        x, y, cw, ch = comp['box']
        pad = 5
        roi = labels_im[max(0, y-pad):min(h, y+ch+pad), 
                        max(0, x-pad):min(w, x+cw+pad)]
        
        unique_nodes = np.unique(roi)
        connected_nodes = [n for n in unique_nodes if n != 0]
        comp_connections.append(connected_nodes)

    return comp_connections

def generate_spice_text(components, connections):
    netlist = ["* Auto-Generated SPICE Netlist"]
    
    for i, comp in enumerate(components):
        label = comp['type']
        name = comp['name']
        nodes = connections[i]
        expected_pins = cfg.get_pin_names(label)
        
        if label == 'ground':
            if nodes: netlist.append(f"* Node {nodes[0]} is GND (0)")
            continue

        # Map nodes to pins
        current_nodes = []
        for j in range(len(expected_pins)):
            if j < len(nodes): current_nodes.append(str(nodes[j]))
            else: current_nodes.append(f"NC_{name}_{j}")

        node_str = " ".join(current_nodes)
        val = "1k" # Placeholder
        if label == 'voltage': val = "5V"
        elif label == 'capacitor': val = "1u"

        netlist.append(f"{name} {node_str} {val}")
        
    return "\n".join(netlist)