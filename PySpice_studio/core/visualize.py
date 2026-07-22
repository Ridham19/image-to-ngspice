import cv2
import numpy as np
import base64

def get_node_color(node_id):
    """Return a unique, vibrant color for each node ID."""
    if node_id == "0":
        return (0, 255, 0)  # Green for Ground / Reference node
    
    # Predefined high-contrast BGR colors
    colors = [
        (255, 0, 255),   # Magenta
        (0, 165, 255),   # Orange
        (255, 255, 0),   # Cyan
        (0, 255, 255),   # Yellow
        (255, 0, 0),     # Blue
        (0, 0, 255),     # Red
        (128, 0, 128),   # Purple
        (0, 128, 128),   # Teal
        (203, 192, 255), # Pink
        (255, 128, 0),   # Light Blue/Azure
        (128, 255, 0),   # Lime Green
    ]
    try:
        idx = int(node_id) - 1
        return colors[idx % len(colors)]
    except ValueError:
        # Fallback for NC or custom node names
        return (120, 120, 120)

def add_panel_header(img, title_text):
    """Add a clean header band to the top of an image panel."""
    h, w = img.shape[:2]
    header_h = 50
    # Create black header block
    header = np.zeros((header_h, w, 3), dtype=np.uint8)
    # Add text
    cv2.putText(
        header, title_text, (15, 32), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
    )
    # Concatenate vertically
    return np.vstack((header, img))

def resize_to_height(img, target_height):
    """Resize image to target height while maintaining aspect ratio."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    target_width = int(w * (target_height / h))
    return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)

def generate_debug_image(original, binary, debug_overlay, detected_comps, connections, pin_anchors):
    """
    Generate a 4-panel debug image and return it as a base64 encoded string.
    """
    h_orig, w_orig = original.shape[:2]

    # Panel 1: Component & Pin Detections
    panel1 = original.copy()
    
    # Draw YOLO boxes and names
    for comp in detected_comps:
        comp_type = comp.get("type", "")
        if comp_type in ["wire", "junction"]:
            continue
            
        box = comp.get("box")
        if box:
            bx, by, bw, bh = map(int, box)
            # Draw bounding box
            cv2.rectangle(panel1, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            
            # Label
            name = comp.get("name", "U")
            val = comp.get("value", "")
            label_str = f"{name} ({comp_type})"
            if val and val != "TEXT_FOUND":
                label_str += f" = {val}"
                
            cv2.putText(
                panel1, label_str, (bx, max(15, by - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA
            )
            
            # If there's an associated text box, draw a faint orange border around it
            if "text_box" in comp:
                tx, ty, tw, th = map(int, comp["text_box"])
                cv2.rectangle(panel1, (tx, ty), (tx + tw, ty + th), (0, 140, 255), 1)

    # Draw pin anchors
    for pin in pin_anchors:
        cv2.circle(panel1, (pin["x"], pin["y"]), 5, (0, 0, 255), -1)  # Red filled dot
        # Tiny label near pin
        cv2.putText(
            panel1, f"p{pin['pin_id']}", (pin["x"] + 6, pin["y"] + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 220), 1, cv2.LINE_AA
        )

    # Panel 2: Raw Edge Detection Mask
    if len(binary.shape) == 2:
        panel2 = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    else:
        panel2 = binary.copy()

    # Panel 3: Isolated Wire Mask
    if len(debug_overlay.shape) == 2:
        panel3 = cv2.cvtColor(debug_overlay, cv2.COLOR_GRAY2BGR)
    else:
        panel3 = debug_overlay.copy()

    # Panel 4: Traced Connection Graph
    panel4 = original.copy()
    panel4 = cv2.addWeighted(panel4, 0.35, np.zeros_like(panel4), 0.65, 0)
    
    # Draw connections (traced wire paths)
    if connections:
        for conn in connections:
            p1 = conn.get("pin1")
            p2 = conn.get("pin2")
            if not p1 or not p2:
                continue
                
            color = (0, 165, 255) # Orange for wire links
            
            pt1 = (int(p1["x"]), int(p1["y"]))
            pt2 = (int(p2["x"]), int(p2["y"]))
            cv2.line(panel4, pt1, pt2, color, 3, cv2.LINE_AA)
            
            # Draw a small visual indicator halfway
            mid_x = (pt1[0] + pt2[0]) // 2
            mid_y = (pt1[1] + pt2[1]) // 2
            cv2.circle(panel4, (mid_x, mid_y), 4, (255, 255, 255), -1)

    # Draw color-coded pin nodes to show matching assignments
    for comp in detected_comps:
        comp_type = comp.get("type", "")
        if comp_type in ["wire", "junction"]:
            continue
            
        comp_nodes = comp.get("nodes", [])
        
        # We need to filter pin_anchors for this specific component
        comp_idx = detected_comps.index(comp)
        comp_pins = [p for p in pin_anchors if p["comp_idx"] == comp_idx]
        
        for pin, node_id in zip(comp_pins, comp_nodes):
            if node_id == "NC":
                cv2.circle(panel4, (pin["x"], pin["y"]), 6, (120, 120, 120), -1)
            else:
                color = get_node_color(node_id)
                cv2.circle(panel4, (pin["x"], pin["y"]), 6, color, -1)
                cv2.circle(panel4, (pin["x"], pin["y"]), 7, (255, 255, 255), 1)
                cv2.putText(
                    panel4, str(node_id), (pin["x"] + 8, pin["y"] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
                )

    # Assemble Final Side-by-Side Image
    target_h = 600
    p1_res = resize_to_height(panel1, target_h)
    p2_res = resize_to_height(panel2, target_h)
    p3_res = resize_to_height(panel3, target_h)
    p4_res = resize_to_height(panel4, target_h)
    
    p1_final = add_panel_header(p1_res, "1. Component & Pin Detections")
    p2_final = add_panel_header(p2_res, "2. Raw Edge Detection Mask")
    p3_final = add_panel_header(p3_res, "3. Isolated Wire Mask")
    p4_final = add_panel_header(p4_res, "4. Traced Connection Graph")
    
    top_row = np.hstack((p1_final, p2_final))
    bottom_row = np.hstack((p3_final, p4_final))
    
    # Make widths match if they slightly differ due to aspect ratio math
    if top_row.shape[1] != bottom_row.shape[1]:
        min_w = min(top_row.shape[1], bottom_row.shape[1])
        top_row = top_row[:, :min_w]
        bottom_row = bottom_row[:, :min_w]

    final_image = np.vstack((top_row, bottom_row))
    
    # Encode to base64
    success, encoded_image = cv2.imencode('.jpg', final_image)
    if not success:
        return ""
        
    b64_string = base64.b64encode(encoded_image).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_string}"
