import logging
import cv2
import numpy as np

logger = logging.getLogger("wire_tracer")

def validate_components(components: list[dict]) -> list[str]:
    """
    Validate the component list against the expected schema.
    Returns a list of error strings (empty if valid).
    Checks: required keys exist, bbox values are non-negative ints,
    pin IDs are unique across all components, pin locs are 2-tuples of ints.
    """
    errors = []
    
    if not isinstance(components, list):
        errors.append("Input components must be a list.")
        return errors
        
    seen_pin_ids = set()
    
    for i, comp in enumerate(components):
        if not isinstance(comp, dict):
            errors.append(f"Component at index {i} is not a dictionary.")
            continue
            
        comp_id = comp.get("id")
        if not comp_id:
            errors.append(f"Component at index {i} is missing an 'id'.")
        
        if "label" not in comp:
            errors.append(f"Component '{comp_id}' is missing a 'label'.")
            
        bbox = comp.get("bbox")
        if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            errors.append(f"Component '{comp_id}' has invalid 'bbox'. Must be a 4-tuple.")
        else:
            if not all(isinstance(v, (int, float)) and v >= 0 for v in bbox):
                errors.append(f"Component '{comp_id}' has invalid 'bbox' values. Must be non-negative numbers.")
                
        pins = comp.get("pins")
        if pins is None or not isinstance(pins, list):
            errors.append(f"Component '{comp_id}' has missing or invalid 'pins'. Must be a list.")
        else:
            for j, pin in enumerate(pins):
                if not isinstance(pin, dict):
                    errors.append(f"Component '{comp_id}', pin index {j} is not a dictionary.")
                    continue
                    
                pin_id = pin.get("id")
                if not pin_id:
                    errors.append(f"Component '{comp_id}', pin index {j} is missing 'id'.")
                elif pin_id in seen_pin_ids:
                    errors.append(f"Duplicate pin ID found: '{pin_id}'.")
                else:
                    seen_pin_ids.add(pin_id)
                    
                loc = pin.get("loc")
                if not loc or not isinstance(loc, (list, tuple)) or len(loc) != 2:
                    errors.append(f"Pin '{pin_id}' has invalid 'loc'. Must be a 2-tuple.")
                else:
                    if not all(isinstance(v, (int, float)) for v in loc):
                        errors.append(f"Pin '{pin_id}' has invalid 'loc' values. Must be numbers.")
                        
    return errors

def net_label_to_color_map(num_nets: int) -> dict[int, tuple[int, int, int]]:
    """
    Return a dict mapping net label int -> BGR color tuple.
    Use HSV color space to distribute hues evenly: hue = (label / num_nets) * 180.
    OpenCV uses Hue range [0, 179], Saturation [0, 255], Value [0, 255].
    """
    color_map = {}
    if num_nets <= 0:
        return color_map
        
    for label in range(1, num_nets + 1):
        hue = int((label / num_nets) * 179)
        # Create a 1x1 image in HSV, then convert to BGR
        hsv_pixel = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
        color = bgr_pixel[0, 0]
        color_map[label] = (int(color[0]), int(color[1]), int(color[2]))
        
    return color_map

def draw_debug_overlay(
    image_bgr: np.ndarray,
    components: list[dict],
    label_map: np.ndarray,
    pin_net_map: dict[str, int | None],
) -> np.ndarray:
    """
    Produce a colored debug visualization.
    - Each net gets a distinct color. Color all wire pixels of that net with its color.
    - Draw component bboxes in white.
    - Draw pin locations as small circles, colored by their net.
    - Draw pin IDs as small text labels next to each pin circle.
    - Draw the net index (0, 1, 2...) at the centroid of each net's wire region.
    """
    # Create a copy so we don't modify the input image directly
    overlay = image_bgr.copy()
    
    max_label = int(label_map.max()) if label_map.size > 0 else 0
    color_map = net_label_to_color_map(max_label)
    
    # Color the wire pixels
    for label in range(1, max_label + 1):
        mask = (label_map == label)
        if label in color_map:
            overlay[mask] = color_map[label]
            
            # Draw net index at centroid
            y_coords, x_coords = np.where(mask)
            if len(x_coords) > 0:
                cx, cy = int(np.mean(x_coords)), int(np.mean(y_coords))
                cv2.putText(overlay, f"Net {label}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(overlay, f"Net {label}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                
    for comp in components:
        # Draw bbox in white
        bbox = comp.get("bbox")
        if bbox and len(bbox) == 4:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), 2)
            
        for pin in comp.get("pins", []):
            loc = pin.get("loc")
            pin_id = pin.get("id")
            if loc and pin_id:
                px, py = int(loc[0]), int(loc[1])
                net_label = pin_net_map.get(pin_id)
                
                # Get color for pin
                color = color_map.get(net_label, (128, 128, 128)) if net_label else (128, 128, 128)
                
                # Draw pin circle
                cv2.circle(overlay, (px, py), 5, color, -1)
                cv2.circle(overlay, (px, py), 5, (255, 255, 255), 1)
                
                # Draw pin ID text
                cv2.putText(overlay, pin_id, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                cv2.putText(overlay, pin_id, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return overlay
