import numpy as np
import logging
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger("wire_tracer")

def assign_pins_to_nets(
    label_map: np.ndarray,
    components: list[dict],
    max_search_radius: int = 20,
    strategy: str = "dominant",
) -> dict[str, int | None]:
    """
    For each pin in each component, determine which net label it connects to.

    Search strategy:
        "dominant":  Expand outward from the pin's pixel location in a square ring,
                     ring by ring. At the first ring that finds ANY non-zero label,
                     return the MOST COMMON (dominant) label in that ring.
                     This handles pins that sit at the edge of a wire.

        "nearest":   Return the label of the single closest non-zero pixel
                     (Euclidean distance). More precise but slower.

    Args:
        label_map:          Output of detect_nets_connected_components or nets_from_skeleton_graph
        components:         Full component list with pin locations
        max_search_radius:  If no wire is found within this radius, pin is "floating" (returns None)
        strategy:           "dominant" or "nearest"

    Returns:
        pin_net_map: dict mapping pin_id (str) -> net_label (int) or None if floating
    """
    if not isinstance(label_map, np.ndarray):
        raise ValueError("label_map must be a numpy array.")
        
    pin_net_map = {}
    h, w = label_map.shape
    
    # Pre-compute distance transform if nearest strategy
    if strategy == "nearest":
        # Create a boolean mask of wires
        wire_mask = (label_map > 0)
        if not np.any(wire_mask):
            # No wires at all
            for comp in components:
                for pin in comp.get("pins", []):
                    pin_id = pin.get("id")
                    if pin_id:
                        pin_net_map[pin_id] = None
            return pin_net_map
            
        # distance_transform_edt computes distance to closest 0. 
        # So we want distance to closest wire pixel, which means we pass the inverted mask (~wire_mask).
        # indices return the exact coordinates of the closest wire pixel.
        distances, closest_indices = distance_transform_edt(~wire_mask, return_indices=True)

    for comp in components:
        for pin in comp.get("pins", []):
            pin_id = pin.get("id")
            if not pin_id:
                continue
                
            loc = pin.get("loc")
            if not loc or len(loc) != 2:
                pin_net_map[pin_id] = None
                continue
                
            px, py = int(loc[0]), int(loc[1])
            
            # Clamp to bounds
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))
            
            if strategy == "nearest":
                dist = distances[py, px]
                if dist <= max_search_radius:
                    closest_y = closest_indices[0, py, px]
                    closest_x = closest_indices[1, py, px]
                    pin_net_map[pin_id] = int(label_map[closest_y, closest_x])
                else:
                    pin_net_map[pin_id] = None
                    
            elif strategy == "dominant":
                assigned_label = None
                for r in range(max_search_radius + 1):
                    # Square ring boundaries
                    y1 = max(0, py - r)
                    y2 = min(h, py + r + 1)
                    x1 = max(0, px - r)
                    x2 = min(w, px + r + 1)
                    
                    if r == 0:
                        val = label_map[py, px]
                        if val > 0:
                            assigned_label = int(val)
                            break
                    else:
                        # Extract the ring: the square at r MINUS the square at r-1
                        # This can be tricky to index directly as a ring, so we take the region 
                        # and mask out the inner part.
                        region = label_map[y1:y2, x1:x2]
                        # To find just the perimeter of this region, it's easier to check the borders
                        # Top row, Bottom row, Left col, Right col of the region
                        # (Careful with overlaps, unique is better)
                        
                        top = region[0, :] if y1 == py - r else []
                        bottom = region[-1, :] if y2 == py + r + 1 else []
                        left = region[:, 0] if x1 == px - r else []
                        right = region[:, -1] if x2 == px + r + 1 else []
                        
                        ring_vals = np.concatenate([top, bottom, left, right])
                        ring_vals = ring_vals[ring_vals > 0]
                        
                        if len(ring_vals) > 0:
                            # Count occurrences
                            unique, counts = np.unique(ring_vals, return_counts=True)
                            
                            if len(unique) > 1:
                                logger.warning(f"Pin '{pin_id}' at ({px}, {py}) touches multiple nets at radius {r}: {unique}. Choosing dominant.")
                                
                            assigned_label = int(unique[np.argmax(counts)])
                            break
                            
                pin_net_map[pin_id] = assigned_label
            else:
                raise ValueError(f"Unknown pin assignment strategy: {strategy}")
                
    return pin_net_map
