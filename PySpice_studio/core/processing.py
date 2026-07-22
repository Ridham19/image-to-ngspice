"""
core/processing.py -- Wire Extraction & Component Masking
=========================================================
Step 1 -- preprocess_image:
    Dual-path binarization validated on hand-drawn circuit images.
      Path A  ->  Adaptive Gaussian Threshold (handles non-uniform illumination)
      Path B  ->  Canny edge detector (catches faint / faded pen strokes)
    Both maps are OR-merged, then repaired with a 5x5 ellipse morphological
    close that seals pen-skip gaps without bridging adjacent terminal pins.

Step 2 -- separate_layers:
    BBOX collision masking erases component interiors from the binary mask so
    only external wire traces survive.  Structural H/V open-kernel pass
    (40x1 horizontal, 1x40 vertical) then reinforces wire-run continuity
    before the DFS pixel-graph traversal consumes the mask.

Step 3 -- compute_pin_anchors:
    Derives absolute pixel pin positions from YOLO bounding boxes using
    class-specific offset rules and distance-to-wire rotation calibration.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


# ===================================================================
# PIN OFFSET RULES
# Maps component class -> relative pin positions expressed as fractions
# of the bounding-box half-width / half-height.
#
# For a horizontal 2-terminal device (resistor, capacitor, ...):
#   Pin 1 is at the LEFT edge   -> (-1.0, 0.0) x (half_w, half_h)
#   Pin 2 is at the RIGHT edge  -> (+1.0, 0.0) x (half_w, half_h)
#
# For a vertical source:
#   Pin 1 (positive) is at TOP    -> (0.0, -1.0)
#   Pin 2 (negative) is at BOTTOM -> (0.0, +1.0)
#
# BJTs have 3 terminals with non-trivial geometry.
# ===================================================================
PIN_OFFSET_RULES: Dict[str, List[Tuple[float, float]]] = {
    # 2-terminal horizontal devices
    "resistor":       [(-1.0, 0.0), (1.0, 0.0)],
    "capacitor":      [(-1.0, 0.0), (1.0, 0.0)],
    "inductor":       [(-1.0, 0.0), (1.0, 0.0)],
    "diode":          [(-1.0, 0.0), (1.0, 0.0)],

    # Vertical sources (+ on top, - on bottom)
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
    "transistor":     [(-1.0, 0.0), (0.6, -1.0), (0.6, 1.0)],
}

# Component types whose internals should NOT be masked (they are not symbols)
TRANSPARENT_TYPES = {"wire", "junction", "text", "label"}


# -------------------------------------------------------------------
# Pre-built structuring elements (cached once at import time)
# -------------------------------------------------------------------

# 7x7 ellipse -- closes sub-pixel pen-skip gaps without bridging
# closely-spaced component terminals.  Replaces the old 31x31 square
# kernel which was dangerously large for circuit imagery.
_MORPH_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# Structural wire-run kernels validated in experimental scripts.
# full_ckt_test.py: h_k=(20,1), v_k=(1,20)
# remove_wire_then_predict.py: (40,1) / (1,40)
# 40-pixel runs reliably detect long wire runs without confusing stubs.
_H_WIRE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
_V_WIRE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

# Small dilation kernel -- gently thickens isolated wire pixels so the
# DFS adjacency builder can connect near-miss clusters.
_DILATION_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


# ===================================================================
# STEP 1 -- IMAGE PREPROCESSING & DUAL-PATH BINARIZATION
# ===================================================================

def preprocess_image(
    image_source,
    # Path A: Adaptive Gaussian Threshold
    blur_ksize: int = 5,
    adapt_block_size: int = 15,
    adapt_c: int = 8,
    # Path B: Canny Edge Detection
    canny_low: int = 50,
    canny_high: int = 150,
    canny_blur_ksize: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for a hand-drawn circuit image.

    Dual-path binarization strategy:

    Path A -- Adaptive Gaussian Threshold
        Compares each pixel against the locally-weighted neighbourhood mean.
        Handles non-uniform illumination (shadows, pencil fading) that a
        global Canny threshold would miss.  THRESH_BINARY_INV makes dark
        ink on light paper become white foreground.

    Path B -- Canny Edge Detection
        Detects gradient discontinuities after a heavier Gaussian blur.
        Catches faint strokes that the adaptive threshold may binarize
        inconsistently near block boundaries.

    Both maps are combined with bitwise OR.  A 5x5 ellipse morphological
    close seals sub-pixel pen-skip gaps in the merged mask.

    Parameters
    ----------
    image_source : str or np.ndarray
        File path (str) or an already-loaded BGR image matrix.
        Accepts both to avoid redundant disk reads inside the API route.
    blur_ksize : int
        Gaussian blur kernel size for the adaptive threshold path (odd, >= 1).
    adapt_block_size : int
        Neighbourhood side length for adaptive thresholding (odd, >= 3).
    adapt_c : int
        Constant subtracted from the local weighted mean before thresholding.
        Higher values raise the threshold, reducing false wire pixels.
    canny_low, canny_high : int
        Hysteresis thresholds for the Canny edge detector.
    canny_blur_ksize : int
        Gaussian blur kernel size for the Canny path only (must be odd).
        A heavier blur here suppresses high-frequency paper-texture noise.

    Returns
    -------
    original : np.ndarray  -- Original BGR image.
    gray     : np.ndarray  -- Grayscale uint8 image.
    binary   : np.ndarray  -- Binary wire map (uint8, values in {0, 255}).
    """
    # Load / copy image
    if isinstance(image_source, str):
        original = cv2.imread(image_source)
        if original is None:
            raise FileNotFoundError(
                f"Could not load image from path: {image_source!r}"
            )
    else:
        original = image_source.copy()

    # Grayscale conversion
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # ===========================================
    # PATH A -- Adaptive Gaussian Threshold
    #   THRESH_BINARY_INV: dark ink -> white px
    # ===========================================
    _blur_a = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    _blur_a = max(1, _blur_a)

    blurred_a = cv2.GaussianBlur(gray, (_blur_a, _blur_a), 0) if _blur_a > 1 else gray

    adapt_map = cv2.adaptiveThreshold(
        blurred_a,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=adapt_block_size,
        C=adapt_c,
    )

    # ===========================================
    # PATH B -- Canny Edge Detection
    #   Heavier blur suppresses paper texture.
    # ===========================================
    _blur_b = canny_blur_ksize if canny_blur_ksize % 2 == 1 else canny_blur_ksize + 1
    _blur_b = max(1, _blur_b)

    blurred_b = cv2.GaussianBlur(gray, (_blur_b, _blur_b), 0)
    canny_map = cv2.Canny(blurred_b, canny_low, canny_high)

    # ===========================================
    # MERGE + REPAIR
    # ===========================================
    merged = cv2.bitwise_or(adapt_map, canny_map)
    binary = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, _MORPH_CLOSE_KERNEL)

    return original, gray, binary


# ===================================================================
# STEP 2a -- COMPONENT COLLISION MASKING + WIRE CONTINUITY ENHANCEMENT
# ===================================================================

def separate_layers(
    gray: np.ndarray,
    binary: np.ndarray,
    detections: Optional[List[Dict[str, Any]]] = None,
    mask_padding: int = 4,
    enhance_wire_continuity: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Separate wire traces from component symbol internals.

    Phase 1 -- Collision Masking
        Paints black rectangles over every detected component bounding box
        so that internal symbol details (e.g. the zig-zag of a resistor)
        are erased.  Only external wire traces remain.  mask_padding shrinks
        each rectangle slightly inward so wire stubs at component terminals
        are preserved.

    Phase 2 -- Wire Continuity Enhancement (default=True)
        Applies structural 40x1 horizontal and 1x40 vertical open-morphology
        kernels (validated in the experimental scripts) to detect and
        reinforce pure wire runs.  The extracted run pixels are gently
        dilated (3x3) and OR-merged back into the wire mask, repairing
        small discontinuities without introducing false connections.

    Parameters
    ----------
    gray       : Grayscale image (used only for the debug overlay).
    binary     : Binary edge map from preprocess_image.
    detections : List of YOLO detection dicts with 'box': [x, y, w, h].
    mask_padding : Extra pixels to shrink each mask rectangle inward.
    enhance_wire_continuity : Apply H/V structural kernel pass after masking.

    Returns
    -------
    component_mask : Binary mask showing ONLY component interiors.
    wire_mask      : Binary mask showing ONLY external wire traces.
    debug_overlay  : BGR debug image with tinted masked/wire regions.
    """
    h, w = binary.shape[:2]

    # Start with a copy; component regions will be erased from this
    wire_mask = binary.copy()

    # Component-only mask (inverse of what wire_mask keeps)
    component_mask = np.zeros_like(binary)

    # Phase 1: BBOX collision masking
    if detections:
        for det in detections:
            comp_type = det.get("type", "")

            # Skip transparent types -- their pixels are part of the wiring
            if comp_type in TRANSPARENT_TYPES:
                continue

            box = det.get("box")
            if box is None or len(box) < 4:
                continue

            bx, by, bw, bh = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # Shrink inward so wire stubs at terminals are not erased.
            # Use dynamic padding based on component size to avoid erasing long stubs on loose bboxes.
            pad = max(mask_padding, int(min(bw, bh) * 0.15))
            pad = min(pad, 12)
            x1 = max(0, bx + pad)
            y1 = max(0, by + pad)
            x2 = min(w, bx + bw - pad)
            y2 = min(h, by + bh - pad)

            if x2 <= x1 or y2 <= y1:
                x1, y1 = max(0, bx), max(0, by)
                x2, y2 = min(w, bx + bw), min(h, by + bh)

            # Black-out component interior on wire mask
            wire_mask[y1:y2, x1:x2] = 0

            # Record the component area for the debug overlay
            component_mask[y1:y2, x1:x2] = 255

    # Phase 2: Structural wire-run continuity enhancement
    if enhance_wire_continuity and np.any(wire_mask > 0):
        # Detect pure horizontal wire runs in the post-mask wire map
        h_wire_runs = cv2.morphologyEx(
            wire_mask, cv2.MORPH_OPEN, _H_WIRE_KERNEL, iterations=1
        )
        # Detect pure vertical wire runs
        v_wire_runs = cv2.morphologyEx(
            wire_mask, cv2.MORPH_OPEN, _V_WIRE_KERNEL, iterations=1
        )

        wire_runs_combined = cv2.bitwise_or(h_wire_runs, v_wire_runs)

        if np.any(wire_runs_combined > 0):
            # Gently thicken to help DFS connect near-miss clusters
            wire_runs_dilated = cv2.dilate(
                wire_runs_combined, _DILATION_KERNEL, iterations=1
            )
            # OR back -- only adds pixels, never removes existing wire data
            wire_mask = cv2.bitwise_or(wire_mask, wire_runs_dilated)

    # Build debug overlay (grayscale background + colour tints)
    debug_overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    debug_overlay[component_mask > 0] = (0, 0, 180)  # Red tint on masked areas

    wire_vis = np.zeros_like(debug_overlay)
    wire_vis[wire_mask > 0] = (0, 220, 0)            # Green tint on wire pixels
    debug_overlay = cv2.addWeighted(debug_overlay, 0.7, wire_vis, 0.3, 0)

    return component_mask, wire_mask, debug_overlay


# ===================================================================
# STEP 2b -- PIN ANCHOR CALIBRATION
# ===================================================================

def compute_pin_anchors(
    detections: List[Dict[str, Any]],
    wire_mask: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    For each detected component, compute absolute pixel positions of its
    terminal connection pins based on its bounding box, the component class,
    and dynamic wire-distance rotation calibration.

    Parameters
    ----------
    detections : YOLO detection list.
    wire_mask  : Optional binary mask for distance-based rotation calibration.
                 When supplied, each candidate rotation is scored by summing
                 nearest-wire distances for all pins; lowest total wins.

    Returns
    -------
    List of pin-anchor dicts, each with keys:
        comp_idx, comp_name, comp_type, pin_id, x, y
    """
    import math
    anchors: List[Dict[str, Any]] = []

    def get_rotated_offsets(offsets_list, rot_deg):
        """Rotate (fx, fy) offsets for BGR screen coordinates (y-down)."""
        if rot_deg == 0:
            return offsets_list
        elif rot_deg == 90:
            return [(-fy, fx) for fx, fy in offsets_list]
        elif rot_deg == 180:
            return [(-fx, -fy) for fx, fy in offsets_list]
        elif rot_deg == 270:
            return [(fy, -fx) for fx, fy in offsets_list]
        return offsets_list

    def get_nearest_wire_dist(px, py, mask):
        """Return distance from (px, py) to nearest wire pixel within 25px."""
        h_m, w_m = mask.shape[:2]
        if px < 0 or px >= w_m or py < 0 or py >= h_m:
            return 50.0
        if mask[py, px] > 0:
            return 0.0
        r = 40
        y_lo, y_hi = max(0, py - r), min(h_m, py + r + 1)
        x_lo, x_hi = max(0, px - r), min(w_m, px + r + 1)
        roi = mask[y_lo:y_hi, x_lo:x_hi]
        ys, xs = np.where(roi > 0)
        if len(xs) == 0:
            return 50.0
        dists_sq = (xs + x_lo - px) ** 2 + (ys + y_lo - py) ** 2
        return float(math.sqrt(np.min(dists_sq)))

    for idx, det in enumerate(detections):
        comp_type = det.get("type", "")
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

        # Look up offset rules for this class; default to horizontal 2-pin
        offsets = PIN_OFFSET_RULES.get(comp_type, [(-1.0, 0.0), (1.0, 0.0)])

        best_rot = 0

        if "rotation" in det:
            best_rot = det["rotation"]
        elif comp_type == "ground":
            best_rot = 0
        elif wire_mask is not None:
            is_vertical   = bh > bw * 1.15
            is_horizontal = bw > bh * 1.15

            default_is_horizontal = False
            default_is_vertical   = False
            if offsets:
                first_fx, first_fy = offsets[0]
                if abs(first_fx) > abs(first_fy):
                    default_is_horizontal = True
                elif abs(first_fy) > abs(first_fx):
                    default_is_vertical = True

            if default_is_horizontal:
                candidates = [90, 270] if is_vertical else [0, 180]
            elif default_is_vertical:
                candidates = [0, 180] if is_vertical else [90, 270]
            else:
                candidates = [0, 90, 180, 270]

            # Select rotation that minimises total pin-to-wire distance
            min_dist = float("inf")
            for rot in candidates:
                cand_offsets = get_rotated_offsets(offsets, rot)
                total_d = 0.0
                for fx, fy in cand_offsets:
                    px = int(round(cx + fx * half_w))
                    py = int(round(cy + fy * half_h))
                    total_d += get_nearest_wire_dist(px, py, wire_mask)
                if total_d < min_dist:
                    min_dist = total_d
                    best_rot = rot
        else:
            # Fallback: swap axes based on aspect ratio when wire mask unavailable
            is_vertical   = bh > bw * 1.15
            is_horizontal = bw > bh * 1.15

            default_is_horizontal = False
            default_is_vertical   = False
            if offsets:
                first_fx, first_fy = offsets[0]
                if abs(first_fx) > abs(first_fy):
                    default_is_horizontal = True
                elif abs(first_fy) > abs(first_fx):
                    default_is_vertical = True

            swap_axes = (
                (default_is_horizontal and is_vertical) or
                (default_is_vertical   and is_horizontal)
            )
            best_rot = 90 if swap_axes else 0

        # Persist rotation in the detection dict for downstream consumers
        det["rotation"] = best_rot

        # Emit pin anchors
        rotated_offsets = get_rotated_offsets(offsets, best_rot)
        for pin_id, (fx, fy) in enumerate(rotated_offsets):
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
