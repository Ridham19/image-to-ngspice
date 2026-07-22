"""
wire_tracer/masking.py — Component Masking & Wire Continuity Enhancement
========================================================================
Removes component interiors from the binary wire mask, then optionally
applies structural wire-run enhancement to repair small discontinuities.
"""
import logging
import cv2
import numpy as np

logger = logging.getLogger("wire_tracer")

# Pre-built structuring elements (cached at import time)
_DILATION_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


def mask_components(
    binary: np.ndarray,
    components: list[dict],
    pin_pad: int = 6,
    erode_bbox: bool = True,
    enhance_wire_continuity: bool = True,
    h_wire_kernel_len: int = 40,
    v_wire_kernel_len: int = 40,
) -> np.ndarray:
    """
    Remove component interiors from the binary wire mask, then enhance
    wire continuity with structural H/V kernels.

    Phase 1 — Collision Masking:
        For each component bounding box, zeroes out the interior pixels
        MINUS a `pin_pad`-pixel border around the bbox. This border
        preserves the wire pixels that connect to pins at component edges.

    Phase 2 — Wire Continuity Enhancement:
        Applies structural 40×1 horizontal and 1×40 vertical
        open-morphology kernels to detect and reinforce pure wire runs.
        The extracted run pixels are gently dilated (3×3) and OR-merged
        back into the wire mask, repairing small discontinuities without
        introducing false connections.

    Args:
        binary:                 Binary wire mask from preprocessing (255 = wire, 0 = background)
        components:             Component list with 'bbox' fields
        pin_pad:                Pixels to leave unmasked around each bbox edge.
        erode_bbox:             If True, shrink the bbox by pin_pad before zeroing.
        enhance_wire_continuity: Apply H/V structural kernel pass after masking.
        h_wire_kernel_len:      Length of horizontal structural kernel.
        v_wire_kernel_len:      Length of vertical structural kernel.

    Returns:
        wire_mask: Copy of binary with component interiors zeroed out
                   and wire continuity enhanced.
    """
    if not isinstance(binary, np.ndarray):
        raise ValueError("Input binary mask must be a numpy array.")

    wire_mask = binary.copy()
    h, w = wire_mask.shape

    masked_count = 0
    total_area_removed = 0

    # ── Phase 1: Collision Masking ──
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

    logger.debug(f"Masked {masked_count} components. Total wire area removed: {total_area_removed} pixels.")

    # ── Phase 2: Wire Continuity Enhancement ──
    if enhance_wire_continuity and np.any(wire_mask > 0):
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_wire_kernel_len, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_wire_kernel_len))

        # Detect pure horizontal wire runs
        h_wire_runs = cv2.morphologyEx(wire_mask, cv2.MORPH_OPEN, h_kernel, iterations=1)
        # Detect pure vertical wire runs
        v_wire_runs = cv2.morphologyEx(wire_mask, cv2.MORPH_OPEN, v_kernel, iterations=1)

        wire_runs_combined = cv2.bitwise_or(h_wire_runs, v_wire_runs)

        if np.any(wire_runs_combined > 0):
            # Gently thicken to help connect near-miss clusters
            wire_runs_dilated = cv2.dilate(wire_runs_combined, _DILATION_KERNEL, iterations=1)
            # OR back — only adds pixels, never removes existing wire data
            wire_mask = cv2.bitwise_or(wire_mask, wire_runs_dilated)

    return wire_mask
