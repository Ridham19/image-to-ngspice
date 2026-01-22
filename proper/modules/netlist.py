import cv2
import numpy as np
from modules.config import cfg # <--- IMPORT CONFIG

def trace_nodes(wire_mask, detected_components):
    """
    Flood fills the wire mask to find connected nodes.
    Returns: list of connected node IDs for each component.
    """
    h, w = wire_mask.shape
    dilated_wires = cv2.dilate(wire_mask, np.ones((5,5), np.uint8), iterations=1)
    num_labels, labels_im = cv2.connectedComponents(dilated_wires)
    
    comp_connections = [] 
    
    for comp in detected_components:
        x, y, cw, ch = comp['box']
        
        # Look at the perimeter of the component box
        pad = 5
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(w, x+cw+pad), min(h, y+ch+pad)
        
        roi_labels = labels_im[y1:y2, x1:x2]
        
        # Find unique node IDs touching this component
        unique_nodes = np.unique(roi_labels)
        connected_nodes = [n for n in unique_nodes if n != 0]
        
        # TODO: Advanced logic needed here to sort nodes by position 
        # (e.g., Left=Pin1, Right=Pin2, Bottom=Pin3 for BJTs)
        
        comp_connections.append(connected_nodes)

    return comp_connections

def generate_spice_text(components, connections):
    netlist = ["* Auto-Generated SPICE Netlist by Ridham19-AI"]
    
    for i, comp in enumerate(components):
        label = comp['type']
        name = comp['name']
        
        # Get specs from Config
        expected_pins = cfg.get_pin_names(label)
        num_required = len(expected_pins)
        
        nodes = connections[i]
        
        # Handle Ground Special Case
        if label == 'ground':
            if nodes:
                netlist.append(f"* Node {nodes[0]} is GND (0)")
            continue

        # Pad nodes if we found fewer wires than pins (e.g., disconnected leg)
        # or Truncate if we found too many noise blobs
        current_nodes = []
        for j in range(num_required):
            if j < len(nodes):
                current_nodes.append(str(nodes[j]))
            else:
                current_nodes.append(f"NC_{name}_{j}") # Not Connected

        # SPICE Format: <NAME> <NODE1> <NODE2> ... <VALUE>
        node_str = " ".join(current_nodes)
        
        # Default Values
        val = "1k"
        if label == 'capacitor': val = "1u"
        elif label == 'inductor': val = "1m"
        elif label == 'voltage': val = "5V"
        elif label == 'bjt': val = "Q2N2222" # Model name for transistor
        elif label == 'mosfet': val = "BS170"
        
        line = f"{name} {node_str} {val}"
        netlist.append(line)
        
    return "\n".join(netlist)