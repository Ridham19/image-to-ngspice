import logging
import cv2
import numpy as np

logger = logging.getLogger("wire_tracer")

def mask_components(
    binary: np.ndarray,
    components: list[dict],
    pin_pad: int = 6,
    erode_bbox: bool = True,
) -> np.ndarray:
    """
    Remove component interiors from the binary wire mask.

    For each component bounding box, zeroes out the interior pixels MINUS
    a `pin_pad`-pixel border around the bbox. This border preserves the wire
    pixels that connect to pins at the component edges.

    Args:
        binary:     Binary wire mask from preprocessing (255 = wire, 0 = background)
        components: Component list with 'bbox' fields
        pin_pad:    Pixels to leave unmasked around each bbox edge.
                    If a component bbox is smaller than 2*pin_pad in either
                    dimension, skip masking that component and log a warning.
        erode_bbox: If True, shrink the bbox by pin_pad before zeroing.
                    If False, zero the full bbox (use only if pins are guaranteed
                    to be outside bboxes).

    Returns:
        wire_mask: Copy of binary with component interiors zeroed out.
    """
    if not isinstance(binary, np.ndarray):
        raise ValueError("Input binary mask must be a numpy array.")
        
    wire_mask = binary.copy()
    h, w = wire_mask.shape
    
    masked_count = 0
    total_area_removed = 0
    
    for comp in components:
        bbox = comp.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
            
        bx, by, bw, bh = [int(v) for v in bbox]
        
        if erode_bbox:
            if bw <= 2 * pin_pad or bh <= 2 * pin_pad:
                logger.warning(f"Component {comp.get('id', 'unknown')} bbox ({bw}x{bh}) is too small for pin_pad {pin_pad}. Skipping masking.")
                continue
                
            x1 = max(0, bx + pin_pad)
            y1 = max(0, by + pin_pad)
            x2 = min(w, bx + bw - pin_pad)
            y2 = min(h, by + bh - pin_pad)
        else:
            x1 = max(0, bx)
            y1 = max(0, by)
            x2 = min(w, bx + bw)
            y2 = min(h, by + bh)
            
        if x1 >= x2 or y1 >= y2:
            continue
            
        # Count area to be removed
        area_before = np.count_nonzero(wire_mask[y1:y2, x1:x2])
        total_area_removed += area_before
        
        # Zero out the interior
        wire_mask[y1:y2, x1:x2] = 0
        masked_count += 1
        
        # Clean up isolated blobs entirely within the FULL bbox that the pin_pad erosion left behind
        # The region of the full bbox:
        fx1 = max(0, bx)
        fy1 = max(0, by)
        fx2 = min(w, bx + bw)
        fy2 = min(h, by + bh)
        
        if fx1 < fx2 and fy1 < fy2:
            bbox_region = wire_mask[fy1:fy2, fx1:fx2]
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bbox_region, connectivity=8)
            
            # Check if any component is completely enclosed within the padded area
            # (i.e. it doesn't touch the borders of fx1..fx2, fy1..fy2)
            for label_idx in range(1, num_labels):
                blob_x, blob_y, blob_w, blob_h, _ = stats[label_idx]
                if (blob_x > 0 and blob_y > 0 and 
                    blob_x + blob_w < (fx2 - fx1) and 
                    blob_y + blob_h < (fy2 - fy1)):
                    # Blob is completely inside the full bbox, remove it
                    wire_mask[fy1:fy2, fx1:fx2][labels == label_idx] = 0
                    
    logger.debug(f"Masked {masked_count} components. Total wire area removed: {total_area_removed} pixels.")
    
    return wire_mask
