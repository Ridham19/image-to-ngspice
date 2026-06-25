"""
core/processing.py — Wire Extraction & Component Masking
Step 1: Canny edge detection + morphological cleanup to isolate wire layers.
Step 2: BBOX collision masking + pin anchor calibration for YOLO detections.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


# ═══════════════════════════════════════════════════════════════════════
# PIN OFFSET RULES
# Maps component class → relative pin positions expressed as fractions
# of the bounding-box half-width / half-height.
#
# For a horizontal 2-terminal device (resistor, capacitor, …):
#   Pin 1 is at the LEFT edge   → (-1.0, 0.0) × (half_w, half_h)
#   Pin 2 is at the RIGHT edge  → (+1.0, 0.0) × (half_w, half_h)
#
# For a vertical source:
#   Pin 1 (positive) is at TOP    → (0.0, -1.0)
#   Pin 2 (negative) is at BOTTOM → (0.0, +1.0)
#
# BJTs have 3 terminals with non-trivial geometry.
# ═══════════════════════════════════════════════════════════════════════
PIN_OFFSET_RULES: Dict[str, List[Tuple[float, float]]] = {
    # 2-terminal horizontal devices
    "resistor":       [(-1.0, 0.0), (1.0, 0.0)],
    "capacitor":      [(-1.0, 0.0), (1.0, 0.0)],
    "inductor":       [(-1.0, 0.0), (1.0, 0.0)],
    "diode":          [(-1.0, 0.0), (1.0, 0.0)],

    # Vertical sources (+ on top, – on bottom)
    "source":         [(0.0, -1.0), (0.0, 1.0)],
    "voltage_source": [(0.0, -1.0), (0.0, 1.0)],
    "current_source": [(0.0, -1.0), (0.0, 1.0)],
    "ac_source":      [(0.0, -1.0), (0.0, 1.0)],

    # Ground has a single pin at its top edge
    "ground":         [(0.0, -1.0)],

    # BJTs: Base (left-center), Collector (right-top), Emitter (right-bottom)
    "bjt_npn":        [(-1.0, 0.0), (0.6, -1.0), (0.6, 1.0)],
    "bjt_pnp":        [(-1.0, 0.0), (0.6, -1.0), (0.6, 1.0)],
    "bjt":            [(-1.0, 0.0), (0.6, -1.0), (0.6, 1.0)],
}

# Component types whose internals should NOT be masked (they are not symbols)
TRANSPARENT_TYPES = {"wire", "junction", "text", "label"}


# ═══════════════════════════════════════════════════════════════════════
# STEP 1 — IMAGE PREPROCESSING & EDGE DETECTION
# ═══════════════════════════════════════════════════════════════════════

def preprocess_image(
    image_source,
    canny_low: int = 50,
    canny_high: int = 150,
    blur_ksize: int = 5,
    morph_ksize: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for a hand-drawn circuit image.

    Parameters
    ----------
    image_source : str or np.ndarray
        File path (str) or an already-loaded BGR image matrix.
    canny_low, canny_high : int
        Hysteresis thresholds for the Canny edge detector.
    blur_ksize : int
        Size of the Gaussian blur kernel (must be odd).
    morph_ksize : int
        Size of the structuring element for morphological closing.

    Returns
    -------
    original : np.ndarray   — Original BGR image.
    gray     : np.ndarray   — Grayscale uint8 image.
    binary   : np.ndarray   — Binary edge map after Canny + morphological closing.
    """
    # ── Load image ──
    if isinstance(image_source, str):
        original = cv2.imread(image_source)
        if original is None:
            raise FileNotFoundError(
                f"Could not load image from path: {image_source}"
            )
    else:
        original = image_source.copy()

    # ── Grayscale conversion ──
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # ── 5×5 Gaussian Blur to suppress high-frequency paper noise ──
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # ── Canny edge detection ──
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # ── Morphological Closing (Dilation → Erosion) ──
    # Repairs microscopic pen skips / cracks in hand-drawn traces
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_ksize, morph_ksize)
    )
    binary = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return original, gray, binary


# ═══════════════════════════════════════════════════════════════════════
# STEP 2a — COMPONENT COLLISION MASKING
# ═══════════════════════════════════════════════════════════════════════

def separate_layers(
    gray: np.ndarray,
    binary: np.ndarray,
    detections: Optional[List[Dict[str, Any]]] = None,
    mask_padding: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Separate wire traces from component symbol internals.

    Paints black rectangles over every detected component's bounding box
    on the binary edge image so that internal symbol details (e.g. the
    zig-zag of a resistor) are erased.  Only external wire traces remain.

    Parameters
    ----------
    gray       : Grayscale image (for debug overlay).
    binary     : Binary edge map from ``preprocess_image``.
    detections : List of YOLO detection dicts, each containing a
                 ``'box': [x, y, w, h]`` key in top-left format.
    mask_padding : Extra pixels to expand each mask rectangle.

    Returns
    -------
    component_mask : Binary mask showing ONLY component interiors.
    wire_mask      : Binary mask showing ONLY external wire traces.
    debug_overlay  : BGR debug image with masked regions highlighted.
    """
    h, w = binary.shape[:2]

    # Start with a copy — we will erase component regions from this
    wire_mask = binary.copy()

    # Component-only mask (inverse of what we keep in wire_mask)
    component_mask = np.zeros_like(binary)

    if detections:
        for det in detections:
            comp_type = det.get("type", "")

            # Skip transparent types — their pixels are part of the wiring
            if comp_type in TRANSPARENT_TYPES:
                continue

            box = det.get("box")
            if box is None or len(box) < 4:
                continue

            bx, by, bw, bh = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # Shrink the mask slightly inward from the bbox edges so we
            # don't accidentally erase wire stubs at the component terminals
            pad = mask_padding
            x1 = max(0, bx + pad)
            y1 = max(0, by + pad)
            x2 = min(w, bx + bw - pad)
            y2 = min(h, by + bh - pad)

            if x2 <= x1 or y2 <= y1:
                # Bounding box too small after padding — mask the full area
                x1, y1 = max(0, bx), max(0, by)
                x2, y2 = min(w, bx + bw), min(h, by + bh)

            # Black-out component interior on wire mask
            wire_mask[y1:y2, x1:x2] = 0

            # Record the component area
            component_mask[y1:y2, x1:x2] = 255

    # Build a debug overlay (grayscale background + red masked areas)
    debug_overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    debug_overlay[component_mask > 0] = (0, 0, 180)  # Red tint on masked areas
    # Green tint on remaining wire pixels
    wire_vis = np.zeros_like(debug_overlay)
    wire_vis[wire_mask > 0] = (0, 220, 0)
    debug_overlay = cv2.addWeighted(debug_overlay, 0.7, wire_vis, 0.3, 0)

    return component_mask, wire_mask, debug_overlay


# ═══════════════════════════════════════════════════════════════════════
# STEP 2b — PIN ANCHOR CALIBRATION
# ═══════════════════════════════════════════════════════════════════════

def compute_pin_anchors(
    detections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For each detected component, compute absolute pixel positions of its
    terminal connection pins based on its bounding box boundaries and the
    component class.

    Parameters
    ----------
    detections : YOLO detection list.  Each dict must contain:
        - ``'type'`` : str  — component class name
        - ``'box'``  : [x, y, w, h]  — bounding box in top-left format
        - ``'name'`` : str  — component instance name (e.g. "R1")
        - ``'center'`` : (cx, cy)  — bounding box center

    Returns
    -------
    List of pin-anchor dicts::

        [
            {
                "comp_idx":  0,
                "comp_name": "R1",
                "comp_type": "resistor",
                "pin_id":    0,
                "x":         120,
                "y":         240,
            },
            ...
        ]
    """
    anchors: List[Dict[str, Any]] = []

    for idx, det in enumerate(detections):
        comp_type = det.get("type", "")

        # Skip types that don't have electrical terminals
        if comp_type in TRANSPARENT_TYPES:
            continue

        box = det.get("box")
        if box is None or len(box) < 4:
            continue

        bx, by, bw, bh = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        cx = bx + bw / 2.0
        cy = by + bh / 2.0
        half_w = bw / 2.0
        half_h = bh / 2.0

        # Look up offset rules for this class, default to horizontal 2-pin
        offsets = PIN_OFFSET_RULES.get(comp_type, [(-1.0, 0.0), (1.0, 0.0)])

        for pin_id, (fx, fy) in enumerate(offsets):
            pin_x = int(round(cx + fx * half_w))
            pin_y = int(round(cy + fy * half_h))

            anchors.append({
                "comp_idx":  idx,
                "comp_name": det.get("name", f"U{idx}"),
                "comp_type": comp_type,
                "pin_id":    pin_id,
                "x":         pin_x,
                "y":         pin_y,
            })

    return anchors
