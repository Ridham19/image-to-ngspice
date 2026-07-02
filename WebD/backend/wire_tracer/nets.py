import numpy as np
import cv2
import logging
from collections import defaultdict

logger = logging.getLogger("wire_tracer")

def detect_nets_connected_components(
    wire_mask: np.ndarray,
    min_wire_area: int = 50,
) -> tuple[int, np.ndarray]:
    """
    Find connected wire nets using cv2.connectedComponents.

    Args:
        wire_mask:      Binary wire mask (255 = wire, 0 = background)
        min_wire_area:  Minimum pixel area for a blob to be considered a real wire.
                        Blobs smaller than this are likely noise and get label 0.

    Returns:
        num_nets:   Number of distinct wire nets found (excludes background label 0)
        label_map:  np.ndarray, shape (H, W), dtype int32.
                    Each pixel has the integer net label it belongs to (0 = background/noise).
    """
    if not isinstance(wire_mask, np.ndarray):
        raise ValueError("Input wire_mask must be a numpy array.")
        
    num_labels_raw, labels_raw, stats, _ = cv2.connectedComponentsWithStats(wire_mask, connectivity=8)
    
    label_map = np.zeros_like(labels_raw, dtype=np.int32)
    current_new_label = 1
    
    for i in range(1, num_labels_raw):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_wire_area:
            # Re-map valid labels to be contiguous from 1 to N
            label_map[labels_raw == i] = current_new_label
            current_new_label += 1
            
    num_nets = current_new_label - 1
    logger.info(f"Detected {num_nets} distinct nets via connected components (filtered {num_labels_raw - 1 - num_nets} small blobs).")
    
    return num_nets, label_map

def build_netlist(
    pin_net_map: dict[str, int | None],
    include_floating_pins: bool = False,
) -> list[list[str]]:
    """
    Group pins by their net label to produce the final netlist.

    Args:
        pin_net_map:           Output of assign_pins_to_nets
        include_floating_pins: If True, pins with None label appear as single-element nets.

    Returns:
        netlist: list of lists of pin IDs. Each inner list is one net.
                 Sorted: nets are ordered by size descending, then alphabetically within each net.
    """
    if not isinstance(pin_net_map, dict):
        raise ValueError("pin_net_map must be a dictionary.")
        
    net_to_pins = defaultdict(list)
    floating_pins = []
    
    for pin_id, net_label in pin_net_map.items():
        if net_label is None:
            floating_pins.append(pin_id)
        else:
            net_to_pins[net_label].append(pin_id)
            
    netlist = []
    
    # Process connected nets
    for pins in net_to_pins.values():
        if len(pins) > 0:
            # Sort pins alphabetically within each net
            netlist.append(sorted(pins))
            
    if include_floating_pins:
        for pin_id in floating_pins:
            netlist.append([pin_id])
            
    # Sort the netlist: first by number of pins descending, then by the first pin ID
    netlist.sort(key=lambda pins: (-len(pins), pins[0] if pins else ""))
    
    return netlist
